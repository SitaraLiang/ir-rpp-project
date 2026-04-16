import glob
from ir_rpp.pref_eval import util
from ir_rpp.pref_eval.pref_eval import get_measures, get_prefs
from tqdm.notebook import tqdm


# TODO: Give this function a clearer name
def load_dfs(
    dataset,
    metrics=["rpp", "invrpp", "dcgrpp", "ap", "ndcg", "rr"],
    binary_relevance=4,
):
    """returns: summary, df_preference, df_metric"""
    qrels_file = f"../../data-source/qrels/{dataset}/2018.txt"
    run_files = glob.glob(f"../../data-source/runs/{dataset}/2018/*.gz")
    measures = get_measures(metrics, None)
    qrels = util.trec_io.read_qrels(qrels_file, binary_relevance, None)

    runs = {}
    for run_file in tqdm(run_files, "Reading run files"):
        runid, run = util.trec_io.read_run(run_file, qrels, None)
        runs[runid] = run

    return get_prefs(1, runs, measures, True)
