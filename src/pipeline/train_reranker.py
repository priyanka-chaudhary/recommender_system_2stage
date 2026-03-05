from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from ..common.utils import ensure_dir
from ..models.faiss_index import load_index, search
from ..models.reranker import train_lgbm_ranker

def _load_ids(artifacts_dir: Path):
    user_ids = json.loads((artifacts_dir / "mappings" / "user_ids.json").read_text(encoding="utf-8"))
    item_ids = json.loads((artifacts_dir / "mappings" / "item_ids.json").read_text(encoding="utf-8"))
    return user_ids, item_ids

def run(processed_dir: Path, artifacts_dir: Path, top_candidates: int, nprobe: int, reranker_params: dict) -> None:
    ensure_dir(artifacts_dir)
    #test = pd.read_parquet(processed_dir / "test.parquet")
    train = pd.read_parquet(processed_dir / "train.parquet")
    target = train.groupby("user_id").tail(1).copy()

    pop = pd.read_parquet(processed_dir / "popularity.parquet")

    user_ids, item_ids = _load_ids(artifacts_dir)
    user2i = {u:i for i,u in enumerate(user_ids)}
    pop_map = {r["item_id"]: float(r["cnt"]) for _, r in pop.iterrows()}

    user_vecs = np.load(artifacts_dir / "user_vecs.npy").astype("float32")
    index = load_index(artifacts_dir / "faiss.index")

    rows_X, rows_y, groups = [], [], []

    for uid, g in target.groupby("user_id"):
        if uid not in user2i:
            continue
        true_items = set(g["item_id"].tolist())
        uidx = user2i[uid]
        scores, idxs = search(index, user_vecs[uidx:uidx+1], k=top_candidates, nprobe=nprobe)
        scores, idxs = scores[0], idxs[0]
        cand_item_ids = [item_ids[i] for i in idxs]

        X_u, y_u = [], []
        for rank, (it, sc) in enumerate(zip(cand_item_ids, scores), start=1):
            X_u.append([float(sc), float(np.log1p(pop_map.get(it, 0.0))), float(rank)])
            y_u.append(1.0 if it in true_items else 0.0)

        if sum(y_u) == 0:
            continue

        rows_X.extend(X_u)
        rows_y.extend(y_u)
        groups.append(len(y_u))

    if len(groups) == 0:
        raise RuntimeError("No training groups created. Increase top_candidates or max_users.")

    X = np.asarray(rows_X, dtype=np.float32)
    y = np.asarray(rows_y, dtype=np.float32)
    group = np.asarray(groups, dtype=np.int32)

    model = train_lgbm_ranker(X, y, group, params=reranker_params)
    joblib.dump(model, artifacts_dir / "reranker.joblib")
    print(f"Saved reranker to {artifacts_dir / 'reranker.joblib'}")
