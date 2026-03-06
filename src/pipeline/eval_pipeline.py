# from __future__ import annotations
# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# import joblib

# from ..models.faiss_index import load_index, search
# from ..eval.offline_eval import eval_rankings

# def _load_ids(artifacts_dir: Path):
#     user_ids = json.loads((artifacts_dir / "mappings" / "user_ids.json").read_text(encoding="utf-8"))
#     item_ids = json.loads((artifacts_dir / "mappings" / "item_ids.json").read_text(encoding="utf-8"))
#     return user_ids, item_ids

# def _popularity_ranking(processed_dir: Path, test: pd.DataFrame, k: int) -> dict:
#     pop = pd.read_parquet(processed_dir / "popularity.parquet")
#     top = pop["item_id"].head(k).tolist()
#     rows = []
#     for uid, g in test.groupby("user_id"):
#         true_items = set(g["item_id"].tolist())
#         for rank, it in enumerate(top, start=1):
#             rows.append((uid, it, float(k - rank), 1 if it in true_items else 0))
#     df = pd.DataFrame(rows, columns=["user_id","item_id","score","label"])
#     return eval_rankings(df, k)

# def run(processed_dir: Path, artifacts_dir: Path, top_candidates: int, nprobe: int, k: int, save_md: str | None = None) -> None:
#     pop = pd.read_parquet(processed_dir / "popularity.parquet")
#     pop_map = {r["item_id"]: float(r["cnt"]) for _, r in pop.iterrows()}

#     test = pd.read_parquet(processed_dir / "test.parquet")
#     user_ids, item_ids = _load_ids(artifacts_dir)
#     user2i = {u:i for i,u in enumerate(user_ids)}

#     user_vecs = np.load(artifacts_dir / "user_vecs.npy").astype("float32")
#     index = load_index(artifacts_dir / "faiss.index")
#     reranker = joblib.load(artifacts_dir / "reranker.joblib")

#     rows = []
#     for uid, g in test.groupby("user_id"):
#         if uid not in user2i:
#             continue
#         true_items = set(g["item_id"].tolist())
#         uidx = user2i[uid]
#         scores, idxs = search(index, user_vecs[uidx:uidx+1], k=top_candidates, nprobe=nprobe)
#         base_scores, idxs = scores[0], idxs[0]
#         cand_item_ids = [item_ids[i] for i in idxs]

#         #X = np.asarray([[float(sc), 0.0, float(rank)] for rank, sc in enumerate(base_scores, start=1)], dtype=np.float32)
#         X = np.asarray(
#             [[float(sc), float(np.log1p(pop_map.get(it, 0.0))), float(rank)]
#             for rank, (it, sc) in enumerate(zip(cand_item_ids, base_scores), start=1)],
#             dtype=np.float32)

#         rerank_scores = reranker.booster.predict(X)

#         for it, sc in zip(cand_item_ids, rerank_scores):
#             rows.append((uid, it, float(sc), 1 if it in true_items else 0))

#     df = pd.DataFrame(rows, columns=["user_id","item_id","score","label"])
#     res = eval_rankings(df, k)

#     base = _popularity_ranking(processed_dir, test, k)
#     key = f"ndcg@{k}"
#     lift = (res[key] - base[key]) / max(base[key], 1e-12)

#     print("Popularity baseline:", base)
#     print("Two-stage model:", res)
#     print(f"Relative lift vs popularity in {key}: {lift*100:.1f}%")

#     if save_md:
#         md = []
#         md.append("# Results\n")
#         md.append(f"Stored in: `{save_md}`\n")
#         md.append("\n## Metrics\n")
#         md.append(f"| Model | NDCG@{k} | Recall@{k} | MAP@{k} |\n")
#         md.append("|---|---:|---:|---:|\n")
#         md.append(f"| Popularity baseline | {base[f'ndcg@{k}']:.5f} | {base[f'recall@{k}']:.5f} | {base[f'map@{k}']:.5f} |\n")
#         md.append(f"| Two-stage recommender | {res[f'ndcg@{k}']:.5f} | {res[f'recall@{k}']:.5f} | {res[f'map@{k}']:.5f} |\n")
#         md.append(
#             f"| Relative lift | {lift*100:+.1f}% | "
#             f"{((res[f'recall@{k}']-base[f'recall@{k}'])/max(base[f'recall@{k}'],1e-12))*100:+.1f}% | "
#             f"{((res[f'map@{k}']-base[f'map@{k}'])/max(base[f'map@{k}'],1e-12))*100:+.1f}% |\n")

#         from pathlib import Path as _Path
#         out = _Path(save_md)
#         out.parent.mkdir(parents=True, exist_ok=True)
#         out.write_text("".join(md), encoding="utf-8")

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


def _to_ranking_df(user_to_items_scores_labels: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(user_to_items_scores_labels, columns=["user_id", "item_id", "score", "label"])


def _popularity_ranking(processed_dir: Path, test: pd.DataFrame, k: int) -> dict:
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    top = pop["item_id"].head(k).tolist()

    rows = []
    for uid, g in test.groupby("user_id"):
        true_items = set(g["item_id"].tolist())
        for rank, it in enumerate(top, start=1):
            # higher score = higher rank
            rows.append((uid, it, float(k - rank), 1 if it in true_items else 0))

    df = _to_ranking_df(rows)
    return eval_rankings(df, k)


def _two_tower_retrieval_ranking(
    test: pd.DataFrame,
    user2i: dict,
    user_vecs: np.ndarray,
    index,
    item_ids: list[str],
    top_candidates: int,
    nprobe: int,
    k: int,
) -> dict:
    rows = []
    for uid, g in test.groupby("user_id"):
        if uid not in user2i:
            continue
        true_items = set(g["item_id"].tolist())
        uidx = user2i[uid]

        scores, idxs = search(index, user_vecs[uidx:uidx + 1], k=top_candidates, nprobe=nprobe)
        base_scores = scores[0]
        idxs = idxs[0]
        cand_item_ids = [item_ids[i] for i in idxs]

        # Use retrieval scores directly
        for it, sc in zip(cand_item_ids, base_scores):
            rows.append((uid, it, float(sc), 1 if it in true_items else 0))

    df = _to_ranking_df(rows)
    return eval_rankings(df, k)


def _category_popularity_ranking(processed_dir: Path, test: pd.DataFrame, k: int) -> dict | None:
    items_path = processed_dir / "items.parquet"
    train_path = processed_dir / "train.parquet"
    pop_path = processed_dir / "popularity.parquet"

    if not items_path.exists() or not train_path.exists() or not pop_path.exists():
        return None

    items = pd.read_parquet(items_path)
    train = pd.read_parquet(train_path)
    pop = pd.read_parquet(pop_path)

    if "category" not in items.columns:
        # RetailRocket or other datasets without categories
        return None

    # Build maps
    item_to_cat = items.set_index("item_id")["category"].to_dict()

    # User top category based on training clicks
    train_cats = train.copy()
    train_cats["category"] = train_cats["item_id"].map(item_to_cat)
    train_cats = train_cats.dropna(subset=["category"])

    if train_cats.empty:
        return None

    user_cat_counts = (
        train_cats.groupby(["user_id", "category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
    )
    user_top_cat = user_cat_counts.drop_duplicates("user_id")[["user_id", "category"]]
    user_to_top_cat = dict(zip(user_top_cat["user_id"], user_top_cat["category"]))

    # Popular items within each category
    cat_item_pop = (
        train_cats.groupby(["category", "item_id"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["category", "cnt"], ascending=[True, False])
    )

    # Precompute top items per category (as lists)
    cat_to_top_items: dict[str, list[str]] = {}
    for cat, g in cat_item_pop.groupby("category"):
        cat_to_top_items[cat] = g["item_id"].head(k).tolist()

    # Global popular fallback
    global_top = pop["item_id"].head(k).tolist()

    rows = []
    for uid, g in test.groupby("user_id"):
        true_items = set(g["item_id"].tolist())
        cat = user_to_top_cat.get(uid)

        recs = []
        if cat is not None:
            recs = cat_to_top_items.get(cat, [])

        # Fill with global popular if not enough
        if len(recs) < k:
            seen = set(recs)
            for it in global_top:
                if it not in seen:
                    recs.append(it)
                    seen.add(it)
                if len(recs) >= k:
                    break

        for rank, it in enumerate(recs[:k], start=1):
            rows.append((uid, it, float(k - rank), 1 if it in true_items else 0))

    df = _to_ranking_df(rows)
    return eval_rankings(df, k)


def run(
    processed_dir: Path,
    artifacts_dir: Path,
    top_candidates: int,
    nprobe: int,
    k: int,
    save_md: str | None = None,
) -> None:
    # Popularity map for reranker features
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    pop_map = {r["item_id"]: float(r["cnt"]) for _, r in pop.iterrows()}

    test = pd.read_parquet(processed_dir / "test.parquet")

    user_ids, item_ids = _load_ids(artifacts_dir)
    user2i = {u: i for i, u in enumerate(user_ids)}

    user_vecs = np.load(artifacts_dir / "user_vecs.npy").astype("float32")
    index = load_index(artifacts_dir / "faiss.index")
    reranker = joblib.load(artifacts_dir / "reranker.joblib")

    # Baseline 1: Global popularity
    base_pop = _popularity_ranking(processed_dir, test, k)

    # Baseline A: Two-tower retrieval only (no reranker)
    base_tw = _two_tower_retrieval_ranking(
        test=test,
        user2i=user2i,
        user_vecs=user_vecs,
        index=index,
        item_ids=item_ids,
        top_candidates=top_candidates,
        nprobe=nprobe,
        k=k,
    )

    # Baseline B: Category popularity (if categories exist)
    base_cat = _category_popularity_ranking(processed_dir, test, k)

    # Two-stage reranking (your main model)
    rows = []
    for uid, g in test.groupby("user_id"):
        if uid not in user2i:
            continue
        true_items = set(g["item_id"].tolist())
        uidx = user2i[uid]

        scores, idxs = search(index, user_vecs[uidx:uidx + 1], k=top_candidates, nprobe=nprobe)
        base_scores, idxs = scores[0], idxs[0]
        cand_item_ids = [item_ids[i] for i in idxs]

        X = np.asarray(
            [
                [float(sc), float(np.log1p(pop_map.get(it, 0.0))), float(rank)]
                for rank, (it, sc) in enumerate(zip(cand_item_ids, base_scores), start=1)
            ],
            dtype=np.float32,
        )
        rerank_scores = reranker.booster.predict(X)

        for it, sc in zip(cand_item_ids, rerank_scores):
            rows.append((uid, it, float(sc), 1 if it in true_items else 0))

    df = _to_ranking_df(rows)
    res = eval_rankings(df, k)

    # Lifts
    key = f"ndcg@{k}"
    lift_vs_pop = (res[key] - base_pop[key]) / max(base_pop[key], 1e-12)
    lift_vs_tw = (res[key] - base_tw[key]) / max(base_tw[key], 1e-12)
    lift_vs_cat = None if base_cat is None else (res[key] - base_cat[key]) / max(base_cat[key], 1e-12)

    print("Popularity baseline:", base_pop)
    print("Two-tower retrieval baseline:", base_tw)
    if base_cat is None:
        print("Category-popularity baseline: N/A (no categories available)")
    else:
        print("Category-popularity baseline:", base_cat)
    print("Two-stage model:", res)
    print(f"Lift vs popularity in {key}: {lift_vs_pop * 100:.1f}%")
    print(f"Lift vs two-tower in {key}: {lift_vs_tw * 100:.1f}%")
    if lift_vs_cat is not None:
        print(f"Lift vs category-popularity in {key}: {lift_vs_cat * 100:.1f}%")

    if save_md:
        def fmt(m: dict, metric: str) -> str:
            return f"{m.get(metric, float('nan')):.5f}"

        md = []
        md.append("# Results\n")
        md.append(f"Stored in: `{save_md}`\n\n")

        md.append("## Metrics\n\n")
        md.append(f"| Model | NDCG@{k} | Recall@{k} | MAP@{k} |\n")
        md.append("|---|---:|---:|---:|\n")
        md.append(f"| Popularity baseline | {fmt(base_pop, f'ndcg@{k}')} | {fmt(base_pop, f'recall@{k}')} | {fmt(base_pop, f'map@{k}')} |\n")
        md.append(f"| Two-tower retrieval | {fmt(base_tw, f'ndcg@{k}')} | {fmt(base_tw, f'recall@{k}')} | {fmt(base_tw, f'map@{k}')} |\n")
        if base_cat is not None:
            md.append(f"| Category popularity | {fmt(base_cat, f'ndcg@{k}')} | {fmt(base_cat, f'recall@{k}')} | {fmt(base_cat, f'map@{k}')} |\n")
        md.append(f"| Two-stage (reranked) | {fmt(res, f'ndcg@{k}')} | {fmt(res, f'recall@{k}')} | {fmt(res, f'map@{k}')} |\n")

        md.append("\n## Lifts (NDCG)\n\n")
        md.append(f"- Lift vs popularity: {lift_vs_pop * 100:+.1f}%\n")
        md.append(f"- Lift vs two-tower: {lift_vs_tw * 100:+.1f}%\n")
        if lift_vs_cat is not None:
            md.append(f"- Lift vs category popularity: {lift_vs_cat * 100:+.1f}%\n")

        out = Path(save_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(md), encoding="utf-8")