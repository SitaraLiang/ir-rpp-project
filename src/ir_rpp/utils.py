def get_shape(d):
    shape = []
    while isinstance(d, dict):
        shape.append(len(d))
        d = next(iter(d.values()))
    return tuple(shape)

def filter_valid_queries(df_metric, df_pref):
    # queries where the mean score of any metric is 0 
    valid_qids = df_metric.groupby("qid")["map-ia"].mean()
    valid_qids = valid_qids[valid_qids > 0].index.tolist()
    
    df_metric_filtered = df_metric[df_metric["qid"].isin(valid_qids)]
    df_pref_filtered = df_pref[df_pref["qid"].isin(valid_qids)]
    
    return df_metric_filtered, df_pref_filtered
