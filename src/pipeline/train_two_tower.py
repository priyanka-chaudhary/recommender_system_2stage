from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from ..common.utils import set_seed, ensure_dir
from ..models.two_tower import TwoTower

class ImplicitPairs(Dataset):
    def __init__(self, user_idx: np.ndarray, item_idx: np.ndarray, num_items: int, neg_samples: int, seed: int):
        self.user_idx = user_idx.astype(np.int64)
        self.item_idx = item_idx.astype(np.int64)
        self.num_items = int(num_items)
        self.neg_samples = int(neg_samples)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int):
        u = self.user_idx[i]
        pos = self.item_idx[i]
        neg = self.rng.integers(0, self.num_items, size=self.neg_samples, endpoint=False)
        return u, pos, neg

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    diff = pos_scores.unsqueeze(1) - neg_scores
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()

def run(processed_dir: Path, artifacts_dir: Path, dim: int, batch_size: int, lr: float, weight_decay: float, neg_samples: int, epochs: int, seed: int) -> None:
    set_seed(seed)
    ensure_dir(artifacts_dir)

    train = pd.read_parquet(processed_dir / "train.parquet")
    user_ids = train["user_id"].unique()
    item_ids = train["item_id"].unique()
    user2i = {u:i for i,u in enumerate(user_ids)}
    item2i = {it:i for i,it in enumerate(item_ids)}

    u_idx = train["user_id"].map(user2i).to_numpy()
    it_idx = train["item_id"].map(item2i).to_numpy()

    ds = ImplicitPairs(u_idx, it_idx, num_items=len(item_ids), neg_samples=neg_samples, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(num_users=len(user_ids), num_items=len(item_ids), dim=dim).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(1, epochs+1):
        losses = []
        for u, pos, neg in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            u = u.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            opt.zero_grad()
            pos_scores = model(u, pos)
            u_rep = u.unsqueeze(1).expand_as(neg)
            neg_scores = model(u_rep.reshape(-1), neg.reshape(-1)).reshape(neg.shape[0], neg.shape[1])
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep} loss: {float(np.mean(losses)):.4f}")

    torch.save(model.state_dict(), artifacts_dir / "two_tower.pt")
    (artifacts_dir / "mappings").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "mappings" / "user_ids.json").write_text(json.dumps(user_ids.tolist()), encoding="utf-8")
    (artifacts_dir / "mappings" / "item_ids.json").write_text(json.dumps(item_ids.tolist()), encoding="utf-8")

    model.eval()
    with torch.no_grad():
        item_vecs = model.item_vectors().cpu().numpy().astype("float32")
        user_vecs = model.user_vectors().cpu().numpy().astype("float32")
    np.save(artifacts_dir / "item_vecs.npy", item_vecs)
    np.save(artifacts_dir / "user_vecs.npy", user_vecs)

    print(f"Saved model + embeddings to {artifacts_dir}")
