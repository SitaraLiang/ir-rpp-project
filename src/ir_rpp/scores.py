from .pref_eval.pref_eval import get_measures, get_prefs
import random
import pandas as pd


def evaluate_preferences(
    qrels,
    runs,
    measures,
    measure_set=None,  # preferences, all, none
    query_eval_wanted=False,
    nosummary=False,
    query_fraction=1.0,
    label_fraction=1.0,
    label_fraction_policy="random",  # random, pool TODO: no query_fraction_policy, is it taken in order then??
    samples=1,
    # topk=None,
) -> tuple[list, list | pd.Dataframe, list | pd.Dataframe]:
    """Main of pref_eval.py adapted for usage in notebooks. Qrels and runs are loaded separately."""

    measures = get_measures(measures, measure_set)

    for sample in range(samples):
        if query_fraction < 1.0:
            qids = list(qrels.keys())
            num_sample = max(int(len(qids) * query_fraction), 1)
            if num_sample < len(qids):
                qids_to_remove = random.sample(qids, len(qids) - num_sample)
                for qid in qids_to_remove:
                    qrels.pop(qid, None)
        if label_fraction < 1.0:
            if label_fraction_policy == "pool":
                # qrels_pool_frequencies = trec_io.compute_qrel_pool_frequencies(
                #     args.runfiles, qrels
                # )
                raise NotImplementedError(
                    "we will not use pooled label fraction policy"
                )
            for qid in qrels.keys():
                dids = list(qrels[qid].keys())
                num_sample = max(int(len(dids) * label_fraction), 1)
                if num_sample < len(dids):
                    if label_fraction_policy == "pool":
                        # dids_to_remove = qrels_pool_frequencies[qid][num_sample:]
                        raise NotImplementedError(
                            "we will not use pooled label fraction policy"
                        )
                    else:
                        dids_to_remove = random.sample(dids, len(dids) - num_sample)
                    for did in dids_to_remove:
                        qrels[qid].pop(did, None)

        summary, preferences, raw_metrics = get_prefs(
            sample=sample,
            runs=runs,
            measures=measures,
            per_query=query_eval_wanted,
            output_df=False,
        )
        full_summary = []
        if not nosummary:
            for pair_tag, m_prefs in summary.items():
                output_object = {}
                output_object["qid"] = "all"
                output_object["runi"] = pair_tag.split(":")[0]
                output_object["runj"] = pair_tag.split(":")[1]
                output_object["sample"] = sample
                for measure, pref in m_prefs.items():
                    output_object[measure] = pref
                full_summary.append(output_object)

        return full_summary, preferences, raw_metrics
