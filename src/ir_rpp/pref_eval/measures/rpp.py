import math
import sys
from pickle import NONE
from ir_rpp.pref_eval.util.relevance_vector import RelevanceVector
from ir_rpp.pref_eval.measures.util import PositionWeighting


def recall_paired_preference(
    u: RelevanceVector, v: RelevanceVector, weighting: PositionWeighting
) -> float:
    retval: float = 0.0
    m_grade_total: int = 0
    for grade in u.grades:
        uu: list[int] = u.vector(grade)
        vv: list[int] = v.vector(grade)
        m_grade: int = len(uu)
        m_grade_total = m_grade_total + m_grade
        rpp_grade: float = rpp(uu, vv, weighting)
        if rpp_grade is None:
            sys.stderr.write(f"qid: {u.qid}\n")
            return None
        retval = retval + float(m_grade) * rpp_grade
    return retval / float(m_grade_total)


def get_weights(m: int, weighting: PositionWeighting) -> list[float]:
    retval: list[float] = []
    mass: float = 0.0
    for i in range(m):
        v: float = 1.0
        if weighting == PositionWeighting.DCG:
            v = 1.0 / math.log2(i + 2)
        if weighting == PositionWeighting.INVERSE:
            v = 1.0 / float(i + 1)
        retval.append(v)
        mass = mass + v
    return [x / mass for x in retval]


#
# Recall-Paired Preference (RPP)
#
def rpp(x: list[int], y: list[int], weighting: PositionWeighting):
    retval: float = 0.0
    mx: int = len(x)
    my: int = len(y)
    if mx != my:
        sys.stderr.write("rpp: relevant document length mismatch\n")
        sys.stderr.write(f"mx={mx}\n")
        sys.stderr.write(f"my={my}\n")
        return None
    m = mx
    weights: list[float] = get_weights(m, weighting)
    for i in range(m):
        if (x[i] is None) and (y[i] is None):
            break
        elif x[i] is None:
            retval = retval - weights[i]
        elif y[i] is None:
            retval = retval + weights[i]
        elif x[i] < y[i]:
            retval = retval + weights[i]
        elif x[i] > y[i]:
            retval = retval - weights[i]
    return float(retval)


def subtopic_paired_preference(
    u: RelevanceVector, v: RelevanceVector, weighting: PositionWeighting
) -> float:
    # set of all possible subtopics for this request
    subtopics = set()
    for pos in u.positions:
        subtopics.update([st for st in pos.grades.keys() if st > 0])
    for pos in v.positions:
        subtopics.update([st for st in pos.grades.keys() if st > 0])
    
    if not subtopics:
        return 0.0

    total_st_rpp = 0.0
    
    for t in subtopics:
        # rank positions f_{i,t} for the ith relevant item with subtopic t
        uu = [pos.position for pos in u.positions if t in pos.grades and pos.grades[t] > 0]
        vv = [pos.position for pos in v.positions if t in pos.grades and pos.grades[t] > 0]
        
        # fix length + padding with None for missing ranks
        m = max(len(uu), len(vv))
        uu += [None] * (m - len(uu))
        vv += [None] * (m - len(vv))
        
        # equivalent to the rpp function for single topic
        total_st_rpp += rpp(uu, vv, weighting)

    # ATTENTION: The paper defines it as a sum over all possible subtopics for this request
    # here we return the average across subtopics to keep it in the same range as other metrics -> results are better
    return total_st_rpp / len(subtopics)