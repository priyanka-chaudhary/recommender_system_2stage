from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path

@dataclass
class ModelCfg:
    embedding_dim: int = 64
    batch_size: int = 4096
    lr: float = 0.003
    weight_decay: float = 0.0
    negative_samples: int = 20
    seed: int = 42

@dataclass
class RetrievalCfg:
    faiss_nlist: int = 256
    faiss_nprobe: int = 16
    top_candidates: int = 200

@dataclass
class RerankerCfg:
    num_leaves: int = 64
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_data_in_leaf: int = 200

@dataclass
class DataCfg:
    max_users: int = 20000
    min_user_interactions: int = 3
    test_holdout: int = 1

@dataclass
class Config:
    dataset: str
    paths: Paths
    model: ModelCfg
    retrieval: RetrievalCfg
    reranker: RerankerCfg
    data: DataCfg

def load_config(dataset: str) -> Config:
    cfg_path = Path("configs") / f"{dataset}.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = Paths(
        raw_dir=Path(raw["paths"]["raw_dir"]),
        processed_dir=Path(raw["paths"]["processed_dir"]),
        artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
    )
    model = ModelCfg(**raw.get("model", {}))
    retrieval = RetrievalCfg(**raw.get("retrieval", {}))
    reranker = RerankerCfg(**raw.get("reranker", {}))
    data = DataCfg(**raw.get("data", {}))
    return Config(
        dataset=raw["dataset"],
        paths=paths,
        model=model,
        retrieval=retrieval,
        reranker=reranker,
        data=data,
    )
