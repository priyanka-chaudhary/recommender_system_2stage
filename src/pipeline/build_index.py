from __future__ import annotations
from pathlib import Path
import numpy as np
from ..common.utils import ensure_dir
from ..models.faiss_index import build_faiss_ivf, save_index

def run(artifacts_dir: Path, nlist: int, nprobe: int) -> None:
    ensure_dir(artifacts_dir)
    item_vecs = np.load(artifacts_dir / "item_vecs.npy").astype("float32")
    index = build_faiss_ivf(item_vecs, nlist=nlist)
    save_index(index, artifacts_dir / "faiss.index")
    (artifacts_dir / "faiss_params.txt").write_text(f"nlist={nlist}\nnprobe={nprobe}\n", encoding="utf-8")
    print(f"Saved FAISS index to {artifacts_dir / 'faiss.index'}")
