from __future__ import annotations
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
import pandas as pd

from src.common.config import load_config
from src.models.faiss_index import load_index, search

app = FastAPI(title="Two-stage Recommender API")
_cache = {}

def _load_artifacts(dataset: str):
    if dataset in _cache:
        return _cache[dataset]

    cfg = load_config(dataset)
    art = cfg.paths.artifacts_dir

    if not (art / "faiss.index").exists():
        raise FileNotFoundError("Missing artifacts. Train and build index first.")

    user_ids = json.loads((art / "mappings" / "user_ids.json").read_text(encoding="utf-8"))
    item_ids = json.loads((art / "mappings" / "item_ids.json").read_text(encoding="utf-8"))
    user2i = {u: i for i, u in enumerate(user_ids)}

    user_vecs = np.load(art / "user_vecs.npy").astype("float32")
    index = load_index(art / "faiss.index")
    reranker = joblib.load(art / "reranker.joblib")

    pop = pd.read_parquet(cfg.paths.processed_dir / "popularity.parquet")
    pop_map = {r["item_id"]: float(r["cnt"]) for _, r in pop.iterrows()}

    _cache[dataset] = (cfg, user2i, item_ids, user_vecs, index, reranker, pop_map)
    return _cache[dataset]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(dataset: str = Query(...), user_id: str = Query(...), k: int = Query(10, ge=1, le=50)):
    try:
        #cfg, user2i, item_ids, user_vecs, index, reranker = _load_artifacts(dataset)
        cfg, user2i, item_ids, user_vecs, index, reranker, pop_map = _load_artifacts(dataset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if user_id not in user2i:
        raise HTTPException(status_code=404, detail="Unknown user_id for this model.")

    uidx = user2i[user_id]
    top_candidates = cfg.retrieval.top_candidates
    scores, idxs = search(index, user_vecs[uidx:uidx+1], k=top_candidates, nprobe=cfg.retrieval.faiss_nprobe)
    base_scores, idxs = scores[0], idxs[0]
    cand_item_ids = [item_ids[i] for i in idxs]

    #X = np.asarray([[float(sc), 0.0, float(rank)] for rank, sc in enumerate(base_scores, start=1)], dtype=np.float32)
    X = np.asarray(
    [
        [float(sc), float(np.log1p(pop_map.get(it, 0.0))), float(rank)]
        for rank, (it, sc) in enumerate(zip(cand_item_ids, base_scores), start=1)
    ],
    dtype=np.float32)

    rerank_scores = reranker.booster.predict(X)
    order = np.argsort(-rerank_scores)[:k]
    recs = [{"item_id": cand_item_ids[i], "score": float(rerank_scores[i])} for i in order]
    return {"dataset": dataset, "user_id": user_id, "k": k, "recommendations": recs}
