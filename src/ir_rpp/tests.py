from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import pandas as pd


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
    reject, pvals_corrected, _, _ = multipletests(df_ttest["pval"], method="bonferroni")
    df_ttest["pval_bonferroni"] = pvals_corrected
    df_ttest["significant"] = reject

    df_summary = round(
        df_ttest.groupby("metric", sort=False)["significant"].mean().to_frame().T * 100,
        2,
    )

    return df_ttest, df_summary

def run_tukeys_hsd_test(df_preference, metrics=["rpp", "invrpp", "dcgrpp", "ap", "ndcg", "rr"]):
    # TODO: Figure out how they implemented this test
    pass