from __future__ import annotations
import numpy as np

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float((rels * discounts).sum())

def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    ideal = np.sort(rels)[::-1]
    denom = dcg_at_k(ideal, k)
    if denom == 0.0:
        return 0.0
    return dcg_at_k(rels, k) / denom

def recall_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    return float(rels.sum() > 0)

def map_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.sum() == 0:
        return 0.0
    precisions = []
    hit = 0
    for i, r in enumerate(rels, start=1):
        if r:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions))
