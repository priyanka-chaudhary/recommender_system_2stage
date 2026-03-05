from __future__ import annotations
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_mind(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = raw_dir / "news.tsv"
    beh_path = raw_dir / "behaviors.tsv"
    if not news_path.exists() or not beh_path.exists():
        raise FileNotFoundError("Expected news.tsv and behaviors.tsv in data/raw/mind/")
    news_cols = ["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"]
    beh_cols = ["impression_id","user_id","time","history","impressions"]
    news = pd.read_csv(news_path, sep="\t", names=news_cols, header=None)
    beh = pd.read_csv(beh_path, sep="\t", names=beh_cols, header=None)
    return news, beh

def build_interactions(news: pd.DataFrame, beh: pd.DataFrame, max_users: int, min_user_interactions: int):
    rows = []
    users = beh["user_id"].dropna().unique()
    users = users[:max_users] if len(users) > max_users else users
    beh2 = beh[beh["user_id"].isin(users)].copy()

    for _, r in tqdm(beh2.iterrows(), total=len(beh2), desc="Parsing behaviors"):
        uid = r["user_id"]
        ts = r["time"]
        imps = str(r["impressions"]).split()
        for imp in imps:
            if "-" not in imp:
                continue
            nid, lab = imp.split("-")
            if int(lab) == 1:
                rows.append((uid, nid, ts))

    pos = pd.DataFrame(rows, columns=["user_id","item_id","timestamp"])
    counts = pos.groupby("user_id")["item_id"].size()
    good_users = counts[counts >= min_user_interactions].index
    pos = pos[pos["user_id"].isin(good_users)].copy()

    item_meta = news[["news_id","category","subcategory","title"]].rename(columns={"news_id":"item_id"})
    return pos, item_meta
