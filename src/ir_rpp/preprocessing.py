import glob
import pandas as pd
from pathlib import Path
from ir_rpp.pref_eval import util
from ir_rpp.pref_eval.pref_eval import get_measures, get_prefs
from tqdm import tqdm  # TODO: idk why tqdm.notebook stopped working


DATA_SOURCE_BASE = Path("../../data-source")

DATASETS = {
    "core": [2017, 2018],
    "deep-docs": [2019, 2020],
    "deep-pass": [2019, 2020],
    "web": [2009, 2010, 2011, 2012, 2013, 2014],
    "robust": [2004],
    "ml-1M": [2018],
    "libraryThing": [2018],
    "beerAdvocate": [2018],
}


def get_paths(dataset: str, year: int | None = None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    years = DATASETS[dataset]

    if year is None:
        if len(years) != 1:
            raise ValueError(f"Dataset '{dataset}' requires a year: {years}")
        year = years[0]

    if year not in years:
        raise ValueError(f"Invalid year {year} for dataset '{dataset}'")

    runs_path = DATA_SOURCE_BASE / "runs" / dataset / str(year)
    qrels_path = DATA_SOURCE_BASE / "qrels" / dataset / f"{year}.txt"

    return runs_path, qrels_path


def load_labels_and_runs(
    dataset: str,
    year: int | None = None,
    binary_relevance=None,
):
    """returns: summary, df_preference, df_metric"""
    runs_path, qrels_file = get_paths(dataset, year)
    run_files = glob.glob(str(runs_path / "*.gz"))
    qrels = util.trec_io.read_qrels(qrels_file, binary_relevance, None)

    runs = {}
    for run_file in tqdm(run_files, desc="Reading run files"):
        runid, run = util.trec_io.read_run(run_file, qrels, None)
        runs[runid] = run

    return qrels, runs


# TODO: Give this function a clearer name
def load_dfs(
    dataset: str,
    year: int | None = None,
    metrics: list | None = None,
    binary_relevance=4,
    per_query=True,
):
    """returns: summary, df_preference, df_metric"""
    if metrics is None:
        metrics = ["rpp", "invrpp", "dcgrpp", "ap", "ndcg", "rr"]

    _, runs = load_labels_and_runs(dataset, year, binary_relevance)
    measures = get_measures(metrics, None)
    return get_prefs(1, runs, measures, per_query)


def dataset_summary(qrels, runs):
    """Reproduce a line from Table 1"""
    return pd.Series(
        {
            "requests": len(qrels),
            "runs": len(runs),
            "rel/request": sum([len(val) for val in qrels.values()]) / len(qrels),
            "subtopics/request": "todo",
        }
    )
