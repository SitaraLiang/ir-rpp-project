from itertools import combinations
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from scipy.stats import kendalltau
from .scores import get_ordering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .scores import ALL_METRICS, EXTENDED_METRICS
from .scores import evaluate_preferences, aggregate_preferences


def run_ttests(df_preference, metrics=["rpp", "dcgrpp", "invrpp", "ap", "ndcg", "rr"]):
    if not isinstance(df_preference, pd.DataFrame):
        df_preference = pd.DataFrame(df_preference)

    groups = df_preference.groupby(["runi", "runj"])
    results = []
    for (runi, runj), g in groups:
        for metric in metrics:
            vals = g[metric].dropna()

            if len(vals) > 1:
                tstat, pval = ttest_1samp(vals, 0.0)

                results.append(
                    {
                        "runi": runi,
                        "runj": runj,
                        "metric": metric,
                        "tstat": tstat,
                        "pval": pval,
                        "n": len(vals),
                    }
                )

    df_ttest = pd.DataFrame(results)

    # Bonferroni correction per metric
    for metric, group in df_ttest.groupby(["metric"]):
        reject, pvals_corrected, _, _ = multipletests(
            group["pval"], method="bonferroni", alpha=0.05
        )
        df_ttest.loc[group.index, "pval_bonferroni"] = pvals_corrected
        df_ttest.loc[group.index, "significant"] = reject

    df_summary = round(
        df_ttest.groupby("metric", sort=False)["significant"].mean().to_frame().T * 100,
        2,
    )

    return df_ttest, df_summary


def run_tukeys_hsd_test(
    df_preference,
    metrics=["rpp", "dcgrpp", "invrpp", "ap", "ndcg", "rr"],
    n_permutations=2000,
    alpha=0.05,
    random_state=None,
    show_progress=True,
):

    if not isinstance(df_preference, pd.DataFrame):
        df_preference = pd.DataFrame(df_preference)

    all_results = []
    all_runs = sorted(
        set(df_preference["runi"].unique()) | set(df_preference["runj"].unique())
    )
    queries = sorted(df_preference["qid"].unique())

    n_runs = len(all_runs)
    n_querys = len(queries)
    run_to_idx = {run: idx for idx, run in enumerate(all_runs)}
    query_to_idx = {query: idx for idx, query in enumerate(queries)}

    for metric in metrics:
        if metric not in df_preference.columns:
            continue
        data_matrix = build_data_matrix(
            df_preference, metric, query_to_idx, run_to_idx, n_querys, n_runs
        )
        results = randomized_tukey_hsd(
            data_matrix, all_runs, n_permutations, alpha, metric
        )
        all_results.append(results)

    df_results = pd.concat(all_results, ignore_index=True)
    df_summary = round(
        df_results.groupby("metric", sort=False)["significant"].mean().to_frame().T
        * 100,
        2,
    )
    return df_results, df_summary


def build_data_matrix(
    df_preference, metric, query_to_idx, run_to_idx, n_queries, n_runs
):
    """
    Average preference described in original paper.
    NEED TO CHECK LATER
    """

    data_matrix = np.full((n_queries, n_runs), np.nan)
    counts = np.zeros((n_queries, n_runs))

    for _, row in tqdm(
        df_preference.iterrows(), total=len(df_preference), desc="Building Data Matrix"
    ):
        t_idx = query_to_idx[row["qid"]]
        i_idx = run_to_idx[row["runi"]]
        j_idx = run_to_idx[row["runj"]]
        val = row[metric]

        if not pd.isna(val):
            if np.isnan(data_matrix[t_idx, i_idx]):
                data_matrix[t_idx, i_idx] = 0
            if np.isnan(data_matrix[t_idx, j_idx]):
                data_matrix[t_idx, j_idx] = 0

            data_matrix[t_idx, i_idx] += val
            data_matrix[t_idx, j_idx] -= val
            counts[t_idx, i_idx] += 1
            counts[t_idx, j_idx] += 1

    # Normalize to get the average relative preference per run
    with np.errstate(divide="ignore", invalid="ignore"):
        data_matrix = np.where(counts > 0, data_matrix / counts, np.nan)

    return data_matrix


def randomized_tukey_hsd(data_matrix, run_names, B=2000, alpha=0.05, metric_name=""):
    """
    Reproduces Carterette's Randomized Tukey HSD.

    Input:
        data_matrix: (n_queries, m_runs) numpy array
        run_names: list of strings (m_runs)
        B: number of permutation trials
        alpha: significance level
    """
    n, m = data_matrix.shape
    pairs = list(combinations(range(m), 2))

    # observed mean differences for all pairs
    # X_bar is a vector of length m
    observed_means = np.nanmean(data_matrix, axis=0)

    q_star_distribution = np.zeros(B)

    # for each trial in 1 to B
    for k in tqdm(range(B), desc="Permutation Trials for " + metric_name):
        # initialize n × m matrix X*
        permuted_X = data_matrix.copy()

        # for each query q do
        for q in range(n):
            # permutation of values in row q of X*
            row = permuted_X[q, :]
            # avoid NaN values
            mask = ~np.isnan(row)
            vals = row[mask]
            np.random.shuffle(vals)
            permuted_X[q, mask] = vals

        # Step 6: Calculate the range of the column means
        perm_means = np.nanmean(permuted_X, axis=0)
        q_star = np.nanmax(perm_means) - np.nanmin(perm_means)
        q_star_distribution[k] = q_star

    # 7-9: Calculate p-values for every pair of runs based on observed mean differences
    results = []
    for i, j in tqdm(pairs, desc="Comparing Pairs"):
        # Step 8: Absolute difference between run i and run j
        diff = np.abs(observed_means[i] - observed_means[j])

        # p-value: probability that a random range q* is greater than our diff
        n_extreme = np.sum(q_star_distribution >= diff)
        p_val = n_extreme / B

        results.append(
            {
                "runi": run_names[i],
                "runj": run_names[j],
                "metric": metric_name,
                "diff": observed_means[i] - observed_means[j],
                "pval": p_val,
                "significant": p_val < alpha,
            }
        )

    return pd.DataFrame(results)


def run_kendal_tau(a: list[str], b: list[str]):
    rank_a = {item: i for i, item in enumerate(a)}
    b_as_ranks = [rank_a[item] for item in b]
    a_ranks = list(range(len(a)))
    tau, p_value = kendalltau(a_ranks, b_as_ranks)
    return tau, p_value


def run_tau_ordering_comparison(
    system_orderings, system_orderings_by_query=None, query_id=None
):
    """Reproduce a part Table 2a (need to average values across all datasets)"""
    df_tau_between_metrics = pd.DataFrame(
        columns=["invrpp", "rpp", "dcgrpp", "ap", "ndcg"],
        index=["rr", "invrpp", "rpp", "dcgrpp", "ap"],
    )
    metrics_tau = ("rr", "invrpp", "rpp", "dcgrpp", "ap", "ndcg")
    for i, metric1 in enumerate(metrics_tau):
        for metric2 in metrics_tau[i + 1 :]:
            if query_id is None:
                tau, p_value = run_kendal_tau(
                    get_ordering(system_orderings, metric1),
                    get_ordering(system_orderings, metric2),
                )
            else:
                assert system_orderings_by_query is not None, "need orderings by query"
                tau, p_value = run_kendal_tau(
                    get_ordering(system_orderings_by_query[query_id], metric1),
                    get_ordering(system_orderings_by_query[query_id], metric2),
                )
            df_tau_between_metrics.loc[metric1, metric2] = tau
    return df_tau_between_metrics


def plot_metric_correlations(
    preferences, metrics=EXTENDED_METRICS, nb_queries=None, nb_prefs=None
):
    df_preferences = pd.DataFrame(preferences)

    # Sample queries (qids)
    if nb_queries is not None:
        unique_qids = df_preferences["qid"].unique()

        sampled_qids = pd.Series(unique_qids).sample(
            n=min(nb_queries, len(unique_qids)), replace=False, random_state=42
        )

        df_query_level_diffs = df_preferences[df_preferences["qid"].isin(sampled_qids)]
    else:
        df_query_level_diffs = df_preferences

    # Sample preferences (rows)
    if nb_prefs is not None:
        df_query_level_diffs = df_query_level_diffs.sample(
            n=min(nb_prefs, len(df_query_level_diffs)), replace=False, random_state=42
        )

    rpp = df_query_level_diffs["rpp"]
    for metric_name in df_query_level_diffs.columns:
        metric = df_query_level_diffs[metric_name]
        if "rpp" not in metric_name and metric_name in metrics:
            do_agree = np.sign(rpp) == np.sign(metric)
            fig, ax = plt.subplots()

            ax.scatter(
                rpp, metric, c=["black" if agree else "red" for agree in do_agree], s=1
            )
            ax.vlines(x=0, ymin=-1, ymax=1, color="black", linewidth=0.3)
            ax.hlines(y=0, xmin=-1, xmax=1, color="black", linewidth=0.3)
            ax.set_xlabel("RPP")
            ax.set_ylabel("$\\Delta$" + metric_name.upper())

            ax.set_title(
                metric_name.upper()
                + f" (r={np.corrcoef(rpp, metric)[0, 1]:.2f}; sa={sum(do_agree) / len(do_agree):.2f})"
            )
        plt.show()


def run_tau_missing_queries(
    pref_eval_output,
    system_orderings,
    num_samples=10,
    query_fractions=np.linspace(0.05, 0.95, 20),
):
    missing_queries_orderings_by_sample = {}
    for query_fraction in tqdm(query_fractions):
        _, missing_queries_orderings_by_sample[float(query_fraction)] = (
            aggregate_preferences(
                pref_eval_output=pref_eval_output,
                query_eval_wanted=False,
                query_fraction=query_fraction,
                num_samples=num_samples,
            )
        )
    missing_queries_tau_mean = {}
    for metric in ALL_METRICS:
        missing_queries_tau_mean[metric] = []
        for query_fraction in query_fractions:
            tau_mean = 0
            n_samples = 0
            for missing_queries_orderings in missing_queries_orderings_by_sample[
                float(query_fraction)
            ]:
                tau, _ = run_kendal_tau(
                    get_ordering(missing_queries_orderings, metric),
                    get_ordering(system_orderings, metric),
                )
                tau_mean += tau
                n_samples += 1
            tau_mean /= n_samples
            missing_queries_tau_mean[metric].append(tau_mean)
    return missing_queries_tau_mean


METRIC_STYLES = {
    "rpp": {"color": "black", "linestyle": "-"},
    "dcgrpp": {"color": "black", "linestyle": "--"},
    "invrpp": {"color": "black", "linestyle": ":"},
    "ndcg": {"color": "red", "linestyle": "-"},
    "ap": {"color": "blue", "linestyle": "-"},
    "rr": {"color": "green", "linestyle": "-"},
}


def plot_missing_queries(
    missing_queries_tau_mean, nb_queries, query_fractions=np.linspace(0.05, 0.95, 20)
):
    for metric in ALL_METRICS:
        style = METRIC_STYLES.get(metric.lower(), {})
        plt.plot(
            [0] + [fraction * nb_queries for fraction in query_fractions],
            [0] + missing_queries_tau_mean[metric],
            label=metric.upper(),
            **style,
        )
        plt.legend()
        plt.xlabel("requests")
        plt.ylabel("$\\tau$")


def run_tau_missing_labels(
    runs, qrels, system_orderings, num_samples, label_fractions=np.linspace(0.1, 0.9, 9)
):

    missing_labels_orderings_by_sample = {}

    for label_fraction in tqdm(label_fractions):
        missing_labels_orderings_by_sample[float(label_fraction)] = []
        for _ in range(num_samples):
            summary, preferences, raw_metrics = evaluate_preferences(
                runs=runs,
                qrels=qrels,
                label_fraction=label_fraction,
                query_eval_wanted=True,
                pbar=False,
            )

            pref_eval_output = summary + preferences + raw_metrics

            _, system_orderings_by_sample = aggregate_preferences(
                pref_eval_output=pref_eval_output,
                query_eval_wanted=True,
            )

            missing_labels_orderings_by_sample[float(label_fraction)].extend(
                system_orderings_by_sample
            )
    missing_labels_tau_mean = {}
    missing_labels_tau_std = {}
    for metric in ALL_METRICS:
        missing_labels_tau_mean[metric] = []
        missing_labels_tau_std[metric] = []
        for label_fraction in label_fractions:
            tau_values = []
            for missing_labels_orderings in missing_labels_orderings_by_sample[
                float(label_fraction)
            ]:
                tau, _ = run_kendal_tau(
                    get_ordering(missing_labels_orderings, metric),
                    get_ordering(system_orderings, metric),
                )
                tau_values.append(tau)
            missing_labels_tau_mean[metric].append(np.mean(tau_values))
            missing_labels_tau_std[metric].append(np.mean(tau_values))
    return missing_labels_tau_mean, missing_labels_tau_std


def plot_missing_labels(
    missing_labels_tau_mean,
    missing_labels_tau_std,
    label_fractions=np.linspace(0.1, 0.9, 9),
    plot_errorbars=True,
):
    for metric in ALL_METRICS:
        style = METRIC_STYLES.get(metric.lower(), {})
        x = [fraction * 100 for fraction in label_fractions] + [100]
        y = missing_labels_tau_mean[metric] + [1]
        yerr = missing_labels_tau_std[metric] + [0]

        if plot_errorbars:
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                label=metric.upper(),
                capsize=3,
                **style,
            )
        else:
            plt.plot(
                x,
                y,
                label=metric.upper(),
                **style,
            )

    plt.xticks(
        ticks=[fraction * 100 for fraction in label_fractions] + [100],
        labels=[str(100 - fraction * 100) for fraction in label_fractions] + ["0"],
    )
    plt.legend()
    plt.ylim((0.3, 1.0))
    plt.xlabel("missing (%)")
    plt.ylabel("$\\tau$")
