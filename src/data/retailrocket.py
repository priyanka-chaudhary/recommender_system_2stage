from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_retailrocket(raw_dir: Path) -> pd.DataFrame:
    events_path = raw_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError("Expected events.csv in data/raw/retailrocket/")
    df = pd.read_csv(events_path)
    df = df.rename(columns={"visitorid":"user_id","itemid":"item_id"})
    return df

def build_interactions(events: pd.DataFrame, max_users: int, min_user_interactions: int):
    events = events[events["event"].isin(["view","addtocart","transaction"])].copy()
    users = events["user_id"].dropna().unique()
    users = users[:max_users] if len(users) > max_users else users
    events = events[events["user_id"].isin(users)].copy()

    inter = events[["user_id","item_id","timestamp"]].copy()
    counts = inter.groupby("user_id")["item_id"].size()
    good_users = counts[counts >= min_user_interactions].index
    inter = inter[inter["user_id"].isin(good_users)].copy()

    item_meta = pd.DataFrame({"item_id": inter["item_id"].unique()})
    return inter, item_meta
