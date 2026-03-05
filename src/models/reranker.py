from __future__ import annotations
import lightgbm as lgb
import numpy as np
from dataclasses import dataclass

@dataclass
class RerankerModel:
    booster: lgb.LGBMRanker
    feature_names: list[str]

def train_lgbm_ranker(X: np.ndarray, y: np.ndarray, group: np.ndarray, params: dict) -> RerankerModel:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=params.get("num_leaves", 64),
        learning_rate=params.get("learning_rate", 0.05),
        n_estimators=params.get("n_estimators", 300),
        min_data_in_leaf=params.get("min_data_in_leaf", 200),
    )
    model.fit(X, y, group=group)
    return RerankerModel(booster=model, feature_names=[f"f{i}" for i in range(X.shape[1])])

def predict(model: RerankerModel, X: np.ndarray) -> np.ndarray:
    return model.booster.predict(X)
