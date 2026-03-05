from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..common.utils import ensure_dir
from ..data.splits import make_last_k_holdout
from ..data.mind import load_mind, build_interactions as build_mind
from ..data.retailrocket import load_retailrocket, build_interactions as build_rr

def run(dataset: str, raw_dir: Path, processed_dir: Path, max_users: int, min_user_interactions: int, holdout_k: int) -> None:
    ensure_dir(processed_dir)

    if dataset == "mind":
        news, beh = load_mind(raw_dir)
        inter, item_meta = build_mind(news, beh, max_users=max_users, min_user_interactions=min_user_interactions)
    elif dataset == "retailrocket":
        events = load_retailrocket(raw_dir)
        inter, item_meta = build_rr(events, max_users=max_users, min_user_interactions=min_user_interactions)
    else:
        raise ValueError("dataset must be mind or retailrocket")

    train, test = make_last_k_holdout(inter, holdout_k=holdout_k)

    train.to_parquet(processed_dir / "train.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    item_meta.to_parquet(processed_dir / "items.parquet", index=False)

    pop = train.groupby("item_id").size().reset_index(name="cnt").sort_values("cnt", ascending=False)
    pop.to_parquet(processed_dir / "popularity.parquet", index=False)

    print(f"Saved processed data to {processed_dir}")
    print(f"Train interactions: {len(train):,}")
    print(f"Test interactions: {len(test):,}")
