from .pref_eval.pref_eval import get_measures, get_prefs
import random
import pandas as pd

from .pref_eval.util.pref_io import (
    read_qids,
    read_prefs,
    read_all_measure_names,
    read_metrics,
)
from .pref_eval.util.pref_io import get_query_rankings_from_preferences
from .pref_eval.util.pref_io import get_query_rankings_from_metrics
from .pref_eval.util.pref_io import Preferences, Metrics, Rankings
from .pref_eval.aggregation import rank_aggregation

from .pref_eval.measures.measures import is_metric

from copy import deepcopy


def subsample_labels(runs, label_fraction):
    runs = deepcopy(runs)
    ...
    return runs


def subsample_queries(runs, query_fraction):
    runs = deepcopy(runs)

    if query_fraction < 1.0:
        qids = list(next(iter(runs.values())).keys())
        num_sample = max(int(len(qids) * query_fraction), 1)
        if num_sample < len(qids):
            qids_to_remove = random.sample(qids, len(qids) - num_sample)
            for system in runs.keys():
                for qid in qids_to_remove:
                    runs[system].pop(qid, None)
    return runs


def evaluate_preferences(
    runs,
    measures,
    measure_set=None,  # preferences, all, none
    query_eval_wanted=False,
    nosummary=False,
    query_fraction=1.0,
    label_fraction=1.0,
    label_fraction_policy="random",  # random, pool TODO: why no query_fraction_policy
    samples=1,
    # topk=None,
) -> tuple[list, list | pd.DataFrame, list | pd.DataFrame]:
    """Main of pref_eval.py adapted for usage in notebooks. Runs are loaded separately."""

    # runs = deepcopy(runs) # NOTE: his code may modify arguments
    measures = get_measures(measures, measure_set)

    full_summary = []
    preferences = []
    raw_metrics = []

    for sample in range(samples):
        # NOTE: it wouldn't work since originally he does it while loading the data
        # - he modifies qrels and then uses t to load runs
        # if query_fraction < 1.0:
        #     qids = list(qrels.keys())
        #     num_sample = max(int(len(qids) * query_fraction), 1)
        #     if num_sample < len(qids):
        #         qids_to_remove = random.sample(qids, len(qids) - num_sample)
        #         for qid in qids_to_remove:
        #             qrels.pop(qid, None)        
        # if label_fraction < 1.0:
        #     if label_fraction_policy == "pool":
        #         # qrels_pool_frequencies = trec_io.compute_qrel_pool_frequencies(
        #         #     args.runfiles, qrels
        #         # )
        #         raise NotImplementedError(
        #             "we will not use pooled label fraction policy"
        #         )
        #     for qid in qrels.keys():
        #         dids = list(qrels[qid].keys())
        #         num_sample = max(int(len(dids) * label_fraction), 1)
        #         if num_sample < len(dids):
        #             if label_fraction_policy == "pool":
        #                 # dids_to_remove = qrels_pool_frequencies[qid][num_sample:]
        #                 raise NotImplementedError(
        #                     "we will not use pooled label fraction policy"
        #                 )
        #             else:
        #                 dids_to_remove = random.sample(dids, len(dids) - num_sample)
        #             for did in dids_to_remove:
        #                 qrels[qid].pop(did, None)
        
        if query_fraction < 1.0:
            runs = subsample_queries(runs, query_fraction)
        
        if label_fraction < 1.0:
            # TODO: implement it...
            raise NotImplementedError
            runs = subsample_labels(runs, label_fraction)

        summary_per_sample, preferences_per_sample, raw_metrics_per_sample = get_prefs(
            sample=sample,
            runs=runs,
            measures=measures,
            per_query=query_eval_wanted,
            output_df=False,
        )
        full_summary_per_sample = []
        if not nosummary:
            for pair_tag, m_prefs in summary_per_sample.items():
                output_object = {}
                output_object["qid"] = "all"
                output_object["runi"] = pair_tag.split(":")[0]
                output_object["runj"] = pair_tag.split(":")[1]
                output_object["sample"] = sample
                for measure, pref in m_prefs.items():
                    output_object[measure] = pref
                full_summary_per_sample.append(output_object)
        full_summary.extend(full_summary_per_sample)
        preferences.extend(preferences_per_sample)
        raw_metrics.extend(raw_metrics_per_sample)

    return full_summary, preferences, raw_metrics


def prepare_prefs(prefs, metrics, current_sample, qids=None):
    retval: Preferences = {}
    for obj in prefs:
        qid: str = obj["qid"]
        sample = obj["sample"]
        linetype = obj["type"] if qid != "all" else None
        if (
            (linetype == "metric")
            or (qid == "all")
            or (sample != current_sample)
            or ((qids is not None) and (qid not in qids))
        ):
            continue
        if qid not in retval:
            retval[qid] = {}
        pair_tag = f"{obj['runi']}:{obj['runj']}"
        for metric in metrics:
            if not is_metric(metric):
                preference: float = obj[metric]
                if metric not in retval[qid]:
                    retval[qid][metric] = {}
                retval[qid][metric][pair_tag] = preference
    return retval


def get_metrics_from_prefs(prefs, metrics, current_sample, qids=None):
    print(prefs)
    retval: Metrics = {}
    for obj in prefs:
        qid: str = obj["qid"]
        sample = obj["sample"]
        linetype = obj["type"] if qid != "all" else None
        if (
            (linetype == "preference")
            or (qid == "all")
            or (sample != current_sample)
            or ((qids is not None) and (qid not in qids))
        ):
            continue
        run = obj["run"]
        if qid not in retval:
            retval[qid] = {}
        for metric in metrics:
            if is_metric(metric):
                meas: float = obj[metric]
                if metric not in retval[qid]:
                    retval[qid][metric] = {}
                retval[qid][metric][run] = meas
    return retval


def aggregate_preferences(
    prefs,
    query_eval_wanted=False,
    nosummary=False,
    query_fraction=1.0,
    num_samples=1,
    measures=[],  # default: all measures in the preferences file
    nb_queries = None # NOTE: added myself, keep only first nb_queries queries
):
    """
    Main of pref_aggregate adapted for usage in notebooks. Preferences are loaded separately.
    Note that it does both aggregation across pairs and queries.
    """
    system_orderings_by_query = []
    system_orderings = {}
    
    qids_to_keep = sorted(list({pref["qid"] for pref in prefs if pref["qid"] != "all"}))[:nb_queries] + ["all"] if nb_queries is not None else None
    
    if qids_to_keep is not None:
        prefs: Preferences = [pref for pref in prefs if pref["qid"] in qids_to_keep]
    
    qids = list({pref["qid"] for pref in prefs}) if (query_fraction < 1.0) else None

    sample_size = (
        max(1, int(len(qids) * query_fraction)) if (query_fraction < 1.0) else None
    )

    for sample in range(num_samples):
        sample_qids = None if qids is None else random.sample(qids, sample_size)

        src_sample = 0 if (query_fraction < 1.0) else sample

        if len(measures) == 0:
            measures = [
                key
                for key in prefs[0].keys()
                if key not in ["qid", "sample", "type", "runi", "runj", "run"]
            ]

        metrics: Metrics = get_metrics_from_prefs(
            prefs, measures, src_sample, sample_qids
        )
        prefs: Preferences = prepare_prefs(prefs, measures, src_sample, sample_qids)

        if sample_qids is None:
            sample_qids = list(prefs.keys())

        p_rankings: Rankings = get_query_rankings_from_preferences(prefs)
        m_rankings: Rankings = get_query_rankings_from_metrics(metrics)
        if query_eval_wanted:
            for qid in sample_qids:
                output_object = {}
                output_object["qid"] = qid
                output_object["sample"] = sample
                for kk, vv in p_rankings[qid].items():
                    output_object[kk] = vv
                for kk, vv in m_rankings[qid].items():
                    output_object[kk] = vv
                system_orderings_by_query.append(output_object)

        if not nosummary:
            output_object = {}
            output_object["qid"] = "all"
            output_object["sample"] = sample
            for measure in measures:
                output_object[measure] = {}
                if is_metric(measure):
                    output_object[measure]["type"] = "metric"
                    avgmeasure = {}
                    numq = float(len(metrics))
                    for qid, v in metrics.items():
                        for run, value in v[measure].items():
                            if run not in avgmeasure:
                                avgmeasure[run] = 0.0
                            avgmeasure[run] += value / numq
                    ranking = [
                        x[1]
                        for x in sorted(
                            [[v, k] for k, v in avgmeasure.items()], reverse=True
                        )
                    ]
                    output_object[measure]["mean"] = ranking
                else:
                    output_object[measure]["type"] = "preference"
                    rankings_metric: list[list[str]] = []
                    for qid, v in p_rankings.items():
                        rankings_metric.append(v[measure])
                    output_object[measure]["mc4"] = rank_aggregation.mc4(
                        rankings_metric
                    )
                    output_object[measure]["borda"] = rank_aggregation.borda(
                        rankings_metric
                    )
            system_orderings = output_object

    return system_orderings_by_query, system_orderings


def get_ordering(system_orderings, metric):
    ordering = system_orderings[metric]
    if isinstance(ordering, dict):
        ordering = ordering.get("mean") or ordering.get("mc4")
    return ordering
