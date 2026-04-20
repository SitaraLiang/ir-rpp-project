from itertools import combinations
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_ttests(df_preference, metrics=["rpp", "invrpp", "dcgrpp", "ap", "ndcg", "rr"]):
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

    # Bonferroni
    reject, pvals_corrected, _, _ = multipletests(df_ttest["pval"], method="bonferroni", alpha=0.05)
    df_ttest["pval_bonferroni"] = pvals_corrected
    df_ttest["significant"] = reject

    df_summary = round(
        df_ttest.groupby("metric", sort=False)["significant"].mean().to_frame().T * 100,
        2,
    )

    return df_ttest, df_summary


def run_tukeys_hsd_test(df_preference, metrics=["rpp", "invrpp", "dcgrpp", "ap", "ndcg", "rr"], 
                        n_permutations=2000, alpha=0.05, random_state=None, show_progress=True):
    
    all_results = []
    all_runs = sorted(set(df_preference['runi'].unique()) | set(df_preference['runj'].unique()))
    queries = sorted(df_preference['qid'].unique())
    
    n_runs = len(all_runs)
    n_querys = len(queries)
    run_to_idx = {run: idx for idx, run in enumerate(all_runs)}
    query_to_idx = {query: idx for idx, query in enumerate(queries)}
    
    for metric in metrics:
        if metric not in df_preference.columns:
            continue
        data_matrix = build_data_matrix(df_preference, metric, query_to_idx, run_to_idx, n_querys, n_runs)
        results = randomized_tukey_hsd(data_matrix, all_runs, n_permutations, alpha, metric)
        all_results.append(results)
    
    df_results = pd.concat(all_results, ignore_index=True)
    df_summary = round(
        df_results.groupby("metric", sort=False)["significant"]
        .mean().to_frame().T * 100, 2
    )
    return df_results, df_summary


def build_data_matrix(df_preference, metric, query_to_idx, run_to_idx, n_queries, n_runs):
    """
    Average preference described in original paper.
    NEED TO CHECK LATER
    """
    
    data_matrix = np.full((n_queries, n_runs), np.nan)
    counts = np.zeros((n_queries, n_runs))
    
    for _, row in tqdm(df_preference.iterrows(), total=len(df_preference), desc="Building Data Matrix"):
        t_idx = query_to_idx[row['qid']]
        i_idx = run_to_idx[row['runi']]
        j_idx = run_to_idx[row['runj']]
        val = row[metric]
        
        if not pd.isna(val):
            if np.isnan(data_matrix[t_idx, i_idx]): data_matrix[t_idx, i_idx] = 0
            if np.isnan(data_matrix[t_idx, j_idx]): data_matrix[t_idx, j_idx] = 0
            
            data_matrix[t_idx, i_idx] += val
            data_matrix[t_idx, j_idx] -= val
            counts[t_idx, i_idx] += 1
            counts[t_idx, j_idx] += 1
            
    # Normalize to get the average relative preference per run
    with np.errstate(divide='ignore', invalid='ignore'):
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
        
        results.append({
            'runi': run_names[i],
            'runj': run_names[j],
            'metric': metric_name,
            'diff': observed_means[i] - observed_means[j],
            'pval': p_val,
            'significant': p_val < alpha
        })
        
    return pd.DataFrame(results)