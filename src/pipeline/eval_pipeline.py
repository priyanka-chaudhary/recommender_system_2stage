from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from ..models.faiss_index import load_index, search
from ..eval.offline_eval import eval_rankings

def _load_ids(artifacts_dir: Path):
    user_ids = json.loads((artifacts_dir / "mappings" / "user_ids.json").read_text(encoding="utf-8"))
    item_ids = json.loads((artifacts_dir / "mappings" / "item_ids.json").read_text(encoding="utf-8"))
    return user_ids, item_ids

def _popularity_ranking(processed_dir: Path, test: pd.DataFrame, k: int) -> dict:
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    top = pop["item_id"].head(k).tolist()
    rows = []
    for uid, g in test.groupby("user_id"):
        true_items = set(g["item_id"].tolist())
        for rank, it in enumerate(top, start=1):
            rows.append((uid, it, float(k - rank), 1 if it in true_items else 0))
    df = pd.DataFrame(rows, columns=["user_id","item_id","score","label"])
    return eval_rankings(df, k)

def run(processed_dir: Path, artifacts_dir: Path, top_candidates: int, nprobe: int, k: int) -> None:
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    pop_map = {r["item_id"]: float(r["cnt"]) for _, r in pop.iterrows()}

    test = pd.read_parquet(processed_dir / "test.parquet")
    user_ids, item_ids = _load_ids(artifacts_dir)
    user2i = {u:i for i,u in enumerate(user_ids)}

    user_vecs = np.load(artifacts_dir / "user_vecs.npy").astype("float32")
    index = load_index(artifacts_dir / "faiss.index")
    reranker = joblib.load(artifacts_dir / "reranker.joblib")

    rows = []
    for uid, g in test.groupby("user_id"):
        if uid not in user2i:
            continue
        true_items = set(g["item_id"].tolist())
        uidx = user2i[uid]
        scores, idxs = search(index, user_vecs[uidx:uidx+1], k=top_candidates, nprobe=nprobe)
        base_scores, idxs = scores[0], idxs[0]
        cand_item_ids = [item_ids[i] for i in idxs]

        #X = np.asarray([[float(sc), 0.0, float(rank)] for rank, sc in enumerate(base_scores, start=1)], dtype=np.float32)
        X = np.asarray(
            [[float(sc), float(np.log1p(pop_map.get(it, 0.0))), float(rank)]
            for rank, (it, sc) in enumerate(zip(cand_item_ids, base_scores), start=1)],
            dtype=np.float32)

        rerank_scores = reranker.booster.predict(X)

        for it, sc in zip(cand_item_ids, rerank_scores):
            rows.append((uid, it, float(sc), 1 if it in true_items else 0))

    df = pd.DataFrame(rows, columns=["user_id","item_id","score","label"])
    res = eval_rankings(df, k)

    base = _popularity_ranking(processed_dir, test, k)
    key = f"ndcg@{k}"
    lift = (res[key] - base[key]) / max(base[key], 1e-12)

    print("Popularity baseline:", base)
    print("Two-stage model:", res)
    print(f"Relative lift vs popularity in {key}: {lift*100:.1f}%")
