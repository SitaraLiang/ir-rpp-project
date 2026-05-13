import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .scores import ALL_METRICS, EXTENDED_METRICS

METRIC_STYLES = {
    "rpp": {"color": "black", "linestyle": "-"},
    "dcgrpp": {"color": "black", "linestyle": "--"},
    "invrpp": {"color": "black", "linestyle": ":"},
    "ndcg": {"color": "red", "linestyle": "-"},
    "ap": {"color": "blue", "linestyle": "-"},
    "rr": {"color": "green", "linestyle": "-"},
}

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
