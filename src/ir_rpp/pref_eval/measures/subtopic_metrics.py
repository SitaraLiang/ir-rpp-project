from ir_rpp.pref_eval.util.relevance_vector import RelevanceVector


def d_strec_at_k(x: RelevanceVector, y: RelevanceVector, k: int = 20) -> float:
    """Calculates the delta in subtopic recall between two systems."""
    return strec_at_k(x, k) - strec_at_k(y, k)

def strec_at_k(rv: RelevanceVector, k: int = 20) -> float:
    # in ndeval.c, subtopics must be the total set from qrels, not just those retrieved
    all_subtopics = set()
    for pos in rv.positions:
        # st_id > 0 in order to match ndeval subtopic indexing
        all_subtopics.update([st for st in pos.grades.keys() if st > 0])
    
    if not all_subtopics:
        return 0.0

    seen_subtopics = set()
    for pos in rv.positions:
        if pos.position is not None and pos.position <= k:
            for st_id, grade in pos.grades.items():
                if st_id > 0 and grade > 0:
                    seen_subtopics.add(st_id)
                
    return len(seen_subtopics) / len(all_subtopics)

def d_map_ia(x: RelevanceVector, y: RelevanceVector) -> float:
    """Calculates the delta in Intent(subtopic)-Aware MAP between two systems."""
    return map_ia(x) - map_ia(y)

def map_ia(rv: RelevanceVector) -> float:
    """Calculates Intent(subtopic)-Aware Mean Average Precision matching ndeval.c logic."""
    all_subtopics = set()
    for pos in rv.positions:
        all_subtopics.update([st for st in pos.grades.keys() if st > 0])
    
    if not all_subtopics:
        return 0.0

    total_ap_ia = 0.0
    
    retrieved = [p for p in rv.positions if p.position is not None]
    retrieved.sort(key=lambda p: p.position)

    for t in all_subtopics:
        subtopic_rel_count = 0
        sum_precision = 0.0
        
        # nrelSub(t) in ndeval.c: total number of relevant documents for subtopic t in the qrels
        total_rel_for_t = sum(1 for p in rv.positions if p.grades.get(t, 0) > 0)
        
        if total_rel_for_t == 0:
            continue
            
        for pos in retrieved:
            # check if the doc is relevant to this subtopic
            if pos.grades.get(t, 0) > 0:
                subtopic_rel_count += 1
                sum_precision += subtopic_rel_count / pos.position
        
        total_ap_ia += (sum_precision / total_rel_for_t)

    return total_ap_ia / len(all_subtopics)


def d_err_ia_at_k(x: RelevanceVector, y: RelevanceVector, k: int = 20, alpha: float = 0.5) -> float:
    """Calculates the delta in ERR-IA between two systems."""
    return err_ia_at_k(x, k, alpha) - err_ia_at_k(y, k, alpha)

def err_ia_at_k(rv: RelevanceVector, k: int = 20, alpha: float = 0.5) -> float:
    all_subtopics = list(set(st for p in rv.positions for st in p.grades.keys() if st > 0))
    m = len(all_subtopics)
    if m == 0:
        return 0.0

    not_satisfied_probs = {st: 1.0 for st in all_subtopics}
    err_total = 0.0
    
    retrieved = [p for p in rv.positions if p.position is not None and p.position <= k]
    retrieved.sort(key=lambda p: p.position)
    
    for i, pos in enumerate(retrieved):
        rank = i + 1
        rank_gain = 0.0
        
        for st in all_subtopics:
            rel_prob = 1.0 if pos.grades.get(st, 0) > 0 else 0.0
            # gain for this subtopic: prob(not satisfied yet) * prob(satisfied now)
            rank_gain += not_satisfied_probs[st] * rel_prob
            not_satisfied_probs[st] *= (1.0 - rel_prob * alpha)
        # sums the average gain across subtopics per rank
        err_total += (rank_gain / m) / rank

    return err_total