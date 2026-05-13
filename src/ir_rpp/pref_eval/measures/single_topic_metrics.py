import math
from pickle import NONE
from ir_rpp.pref_eval.util.relevance_vector import RelevanceVector
from enum import Enum
from ir_rpp.pref_eval.measures.util import PositionWeighting
from ir_rpp.pref_eval.measures.util import GradeType
import numpy as np

#
# METRIC-BASED EVALUATION
#
#
# A. Binary Metrics
#


#
# 1. AP
#
def d_average_precision(x: list[int], y: list[int], k: int = 0) -> float:
    return average_precision(x, k) - average_precision(y, k)


def average_precision(x: list[int], k: int = 0) -> float:
    m: int = len(x)
    if k > 0:
        x = [xi for xi in x if xi != None and xi <= k]
    retval: float = 0.0
    for itr in range(len(x)):
        rl: int = itr + 1
        dx: float = 0.0 if x[itr] is None else 1.0 / float(x[itr])
        retval = retval + float(rl) * dx
    return retval / float(m)


#
# 2. RBP
#
def d_rank_biased_precision(
    x: list[int], y: list[int], k: int = 0, gamma: float = 0.5
) -> float:
    return rank_biased_precision(x, k, gamma) - rank_biased_precision(y, k, gamma)


def d_rank_biased_precision_list(
    x: list[int], y: list[int], k: int = 0, gamma: float = 0.5
) -> list[float]:
    l1 = rank_biased_precision_list(x, k, gamma)
    l2 = rank_biased_precision_list(y, k, gamma)
    return [l1[i] - l2[i] for i in range(len(l1))]


def rank_biased_precision(x: list[int], k: int = 0, gamma: float = 0.5) -> float:
    retval: float = 0.0
    if k > 0:
        x = [xi for xi in x if xi != None and xi <= k]
    m: int = len(x)
    for itr in range(m):
        dx: float = 0.0 if x[itr] is None else gamma ** (x[itr] - 1)
        retval = retval + dx
    # why is there a (1.0 - gamma)? It was here originally
    return (1.0 - gamma) * retval


def rank_biased_precision_list(
    x: list[int], k: int = 0, gamma: float = 0.5
) -> list[float]:
    ret = []
    if k > 0:
        x = [xi for xi in x if xi is not None and xi <= k]
    m: int = len(x)
    for itr in range(m):
        dx: float = 0.0 if x[itr] is None else gamma ** (x[itr] - 1)
        ret.append((1 - gamma) * dx)
    return ret


#
# 3. RR
#
def d_reciprocal_rank(x: list[int], y: list[int]) -> float:
    return reciprocal_rank(x) - reciprocal_rank(y)


def reciprocal_rank(x: list[int], k: int = 0) -> float:
    return 0.0 if x[0] is None or (k > 0 and x[0] > k) else 1.0 / float(x[0])


#
# 4. RelRet
#
def d_number_of_relevant_items_at_k(x: list[int], y: list[int], k: int = 10) -> int:
    return number_of_relevant_items_at_k(x, k) - number_of_relevant_items_at_k(y, k)


def number_of_relevant_items_at_k(x: list[int], k: int = 10) -> int:
    return sum([(i is not None) and (i <= k) for i in x])


#
# 5. Precision
#
def d_precision_at_k(x: list[int], y: list[int], k: int = 10) -> float:
    return float(d_number_of_relevant_items_at_k(x, y, k)) / float(k)


def precision_at_k(x: list[int], k: int = 10) -> float:
    return float(number_of_relevant_items_at_k(x, k)) / float(k)


#
# 6. Recall
#
def d_recall_at_k(x: list[int], y: list[int], k: int = 10) -> float:
    m: int = len(x)
    return float(d_number_of_relevant_items_at_k(x, y, k)) / float(m)


def recall_at_k(x: list[int], k: int = 10) -> float:
    m: int = len(x)
    return float(number_of_relevant_items_at_k(x, k)) / float(m)


#
# 7. R-Precision
#
def d_r_precision(x: list[int], y: list[int]) -> float:
    m: int = len(x)
    return d_precision_at_k(x, y, m)


def r_precision(x: list[int]) -> float:
    m: int = len(x)
    return precision_at_k(x, m)


#
# B. Graded Metrics
#
def dcgopt(x: RelevanceVector, k: int = 0) -> float:
    grade_vector = [xi.grades[0] for xi in x.positions if xi.grades[0] > 0]
    grade_vector.sort(reverse=True)
    if k > 0 and k < len(grade_vector):
        grade_vector = grade_vector[:k]
    m: int = len(grade_vector)
    # XXX: using raw grade as per trec_eval
    return sum([grade_vector[i] / math.log2(i + 2) for i in range(m)])


def dcg(x: RelevanceVector, k: int = 0) -> float:
    gv = [xi for xi in x.positions if (xi.position != None) and (xi.grades[0] > 0)]
    if k > 0:
        gv = [gvi for gvi in gv if gvi.position <= k]
    # XXX: using raw grade as per trec_eval
    return sum([gvi.grades[0] / math.log2(gvi.position + 1) for gvi in gv])


def dcg_list(x: RelevanceVector, k: int = 0) -> list[float]:
    gv = [xi for xi in x.positions if (xi.position != None) and (xi.grades[0] > 0)]
    if k > 0:
        gv = [gvi for gvi in gv if gvi.position <= k]
    # XXX: using raw grade as per trec_eval
    return [gvi.grades[0] / math.log2(gvi.position + 1) for gvi in gv]


def d_ndcg(x: RelevanceVector, y: RelevanceVector, k: int = 0) -> float:
    return (dcg(x, k) - dcg(y, k)) / dcgopt(x, k)


def d_ndcg_list(x: RelevanceVector, y: RelevanceVector, k: int = 0) -> list[float]:
    x_dcg = dcg_list(x, k)
    y_dcg = dcg_list(y, k)
    m = min(len(x_dcg), len(y_dcg))
    ret = []
    for itr in range(m):
        if x_dcg[itr] is None or y_dcg[itr] is None:
            break
        ret.append(x_dcg[itr] - y_dcg[itr])
    return ret


def ndcg(x: RelevanceVector, k: int = 0) -> float:
    return dcg(x, k) / dcgopt(x, k)


#
# Sign
#
def d_sign_list(x: list[int], y: list[int], k: int = 0):
    if k > 0:
        x = [xi for xi in x if xi is not None and xi <= k]
        y = [yi for yi in y if yi is not None and yi <= k]
    m = len(x)
    ret = []
    for itr in range(m):
        if x[itr] is None or y[itr] is None:
            break
        ret.append(np.sign(x[itr] - y[itr]))
    return ret


#
# ASL
#
# TODO: Probably normalize or something
def d_atomized_search_length(x: list[int], y: list[int], k: int = 0) -> float:
    return atomized_search_length(x, k) - average_precision(y, k)


def atomized_search_length(x: list[int], k: int = 0) -> float:
    if k > 0:
        x = [xi for xi in x if xi is not None and xi <= k]
    retval: float = 0.0
    for itr in range(len(x)):
        dx: float = 0.0 if x[itr] is None else x[itr]
        retval += dx
    return retval
