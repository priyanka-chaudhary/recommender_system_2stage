from __future__ import annotations
import pandas as pd

def make_last_k_holdout(interactions: pd.DataFrame, holdout_k: int = 1):
    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id","timestamp"])
    test = df.groupby("user_id").tail(holdout_k).copy()
    train = df.drop(test.index).copy()
    return train, test
