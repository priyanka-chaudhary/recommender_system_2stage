from __future__ import annotations
from pathlib import Path
import numpy as np
import faiss

def build_faiss_ivf(item_matrix: np.ndarray, nlist: int = 256) -> faiss.Index:
    d = item_matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    item_matrix = item_matrix.copy()
    faiss.normalize_L2(item_matrix)
    index.add(item_matrix)
    return index

def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))

def search(index: faiss.Index, user_vecs: np.ndarray, k: int, nprobe: int = 16):
    user_vecs = user_vecs.copy()
    faiss.normalize_L2(user_vecs)
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    scores, idxs = index.search(user_vecs, k)
    return scores, idxs
