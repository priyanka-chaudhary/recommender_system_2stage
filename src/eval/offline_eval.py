from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import ndcg_at_k, recall_at_k, map_at_k

def eval_rankings(df: pd.DataFrame, k: int) -> dict:
    metrics = {"ndcg": [], "recall": [], "map": []}
    for _, g in df.groupby("user_id"):
        g2 = g.sort_values("score", ascending=False)
        rels = g2["label"].to_numpy().astype(int)
        metrics["ndcg"].append(ndcg_at_k(rels, k))
        metrics["recall"].append(recall_at_k(rels, k))
        metrics["map"].append(map_at_k(rels, k))
    return {f"{m}@{k}": float(np.mean(v)) for m, v in metrics.items()}
