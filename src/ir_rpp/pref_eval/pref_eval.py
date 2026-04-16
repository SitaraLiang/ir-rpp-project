#!/usr/bin/env python3.9
import sys
import argparse
import random
from .measures.measures import compute_preference
from .measures.measures import compute_metric
from .measures.measures import is_metric
from .util.relevance_vector import RelevanceVector

# import .util.trec_io
from .util import trec_io
from argparse import RawTextHelpFormatter
import json

from tqdm.notebook import tqdm
import pandas as pd


def get_prefs(
    sample: int, runs: dict[str, trec_io.Run], measures: list[str], per_query: bool
):  # -> dict[str,dict[str,float]]
    runids: list[str] = list(runs.keys())
    qids: list[str] = list(runs[runids[0]].keys())
    numq: int = len(qids)
    retval: dict[str, dict[str, float]] = {}
    pairwise_comparisons = {}
    raw_metrics = []

    for qid in tqdm(qids):
        for i in range(len(runids)):
            runid_i: str = runids[i]
            rv_i: RelevanceVector = runs[runid_i][qid]
            for j in range(i + 1, len(runids)):
                runid_j: str = runids[j]
                rv_j: RelevanceVector = runs[runid_j][qid]
                
                pair = (runid_i, runid_j)
                if pair not in pairwise_comparisons:
                    pairwise_comparisons[pair] = []
                
                # pair_tag = f"{runid_i}:{runid_j}"
                # if pair_tag not in retval:
                #     retval[pair_tag] = {}
                
                output_object = {}
                output_object["qid"] = qid
                output_object["runi"] = runid_i
                output_object["runj"] = runid_j
                output_object["sample"] = sample
                output_object["type"] = "preference"
                for m in measures:
                    pref: float = compute_preference(rv_i, rv_j, m)
                    if pref is None:
                        sys.stderr.write(
                            f"ERROR: qid:{qid} runi:{runid_i} runj:{runid_j} sample:{sample} measure:{m}\n"
                        )
                        u = rv_i.vector()
                        for k in range(len(u)):
                            sys.stderr.write(f"u[{k}]={u[k]}\n")
                        for k in range(len(rv_i.positions)):
                            sys.stderr.write(
                                f"rv_i[{k}].position={rv_i.positions[k].position}\n"
                            )
                            sys.stderr.write(f"rv_i[{k}].did={rv_i.positions[k].did}\n")
                            sys.stderr.write(
                                f"rv_i[{k}].grades[0]={rv_i.positions[k].grades[0]}\n"
                            )
                        v = rv_j.vector()
                        for k in range(len(v)):
                            sys.stderr.write(f"v[{k}]={v[k]}\n")
                        for k in range(len(rv_j.positions)):
                            sys.stderr.write(
                                f"rv_j[{k}].position={rv_j.positions[k].position}\n"
                            )
                            sys.stderr.write(f"rv_j[{k}].did={rv_j.positions[k].did}\n")
                            sys.stderr.write(
                                f"rv_j[{k}].grades[0]={rv_j.positions[k].grades[0]}\n"
                            )
                        sys.exit()
                    output_object[m] = pref
                    # if m not in retval[pair_tag]:
                    #     retval[pair_tag][m] = 0.0
                    # retval[pair_tag][m] = retval[pair_tag][m] + pref / float(numq)
                if per_query:
                    # print(json.dumps(output_object)) # NOTE
                    pairwise_comparisons[pair].append(output_object)

            if per_query:
                output_object = None
                rv: RelevanceVector = runs[runid_i][qid]
                for m in measures:
                    if is_metric(m):
                        if output_object is None:
                            output_object = {}
                            output_object["qid"] = qid
                            output_object["run"] = runid_i
                            output_object["sample"] = sample
                            output_object["type"] = "metric"
                        meas: float = compute_metric(rv, m)
                        output_object[m] = meas
                if output_object is not None:
                    # print(json.dumps(output_object))) # NOTE
                    raw_metrics.append(output_object)
    for pair in pairwise_comparisons:
        pairwise_comparisons[pair] = pd.DataFrame(pairwise_comparisons[pair])
    raw_metrics = pd.DataFrame(raw_metrics)
    # return retval
    return pairwise_comparisons, raw_metrics


def get_measures(m, ms):
    preference_measures = [
        "rpp",
        "invrpp",
        "dcgrpp",
        "lexirecall",
        "lexiprecision",
        "rrlexiprecision",
    ]
    all_measures = [
        "rpp",
        "invrpp",
        "dcgrpp",
        "lexirecall",
        "lexiprecision",
        "rrlexiprecision",
        "ap",
        "rbp",
        "rr",
        "ndcg",
        "rp",
        "p@1",
        "p@10",
        "r@1",
        "r@10",
    ]

    measures: list[str] = m if m is not None else []
    measure_set: str = (ms if ms != "none" else "all") if len(measures) == 0 else ""

    if measure_set == "preferences":
        if len(measures) == 0:
            measures = preference_measures
        else:
            for m in preference_measures:
                if m not in measures:
                    measures.append(m)
    elif measure_set == "all":
        if len(measures) == 0:
            measures = all_measures
        else:
            for m in all_measures:
                if m not in measures:
                    measures.append(m)
    return measures


def prepare_qrels_runs(
    qrels_file,
    runfiles,
    binary_relevance,
    relevance_floor=None,
    topk=None,
    query_fraction=1.0,
    label_fraction=1.0,
    label_fraction_policy="random",
):
    qrels = trec_io.read_qrels(qrels_file, binary_relevance, relevance_floor)
    
    if query_fraction < 1.0:
        qids = list(qrels.keys())
        num_sample = max(int(len(qids) * query_fraction), 1)
        if num_sample < len(qids):
            qids_to_remove = random.sample(qids, len(qids) - num_sample)
            for qid in qids_to_remove:
                qrels.pop(qid, None)

    if label_fraction < 1.0:
        if label_fraction_policy == "pool":
            qrels_pool_frequencies = trec_io.compute_qrel_pool_frequencies(
                runfiles, qrels
            )
        for qid in qrels.keys():
            dids = list(qrels[qid].keys())
            num_sample = max(int(len(dids) * label_fraction), 1)
            if num_sample < len(dids):
                if label_fraction_policy == "pool":
                    dids_to_remove = qrels_pool_frequencies[qid][num_sample:]
                else:
                    dids_to_remove = random.sample(dids, len(dids) - num_sample)
                for did in dids_to_remove:
                    qrels[qid].pop(did, None)

    runs: dict[str, trec_io.Run] = {}

    for runfile in tqdm(runfiles, desc="Reading run files"):
        runid, run = trec_io.read_run(runfile, qrels, topk)
        runs[runid] = run
    return qrels, runs