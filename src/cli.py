from __future__ import annotations
import typer

from src.common.config import load_config
from src.pipeline.prepare_data import run as prepare_run
from src.pipeline.train_two_tower import run as train_two_tower_run
from src.pipeline.build_index import run as build_index_run
from src.pipeline.train_reranker import run as train_reranker_run
from src.pipeline.eval_pipeline import run as eval_run

app = typer.Typer(add_completion=False)

@app.command("prepare-data")
def prepare_data(dataset: str = typer.Option(...), max_users: int = typer.Option(None)):
    cfg = load_config(dataset)
    if max_users is not None:
        cfg.data.max_users = int(max_users)
    prepare_run(
        dataset=cfg.dataset,
        raw_dir=cfg.paths.raw_dir,
        processed_dir=cfg.paths.processed_dir,
        max_users=cfg.data.max_users,
        min_user_interactions=cfg.data.min_user_interactions,
        holdout_k=cfg.data.test_holdout,
    )

@app.command("train-two-tower")
def train_two_tower(dataset: str = typer.Option(...), epochs: int = typer.Option(3)):
    cfg = load_config(dataset)
    train_two_tower_run(
        processed_dir=cfg.paths.processed_dir,
        artifacts_dir=cfg.paths.artifacts_dir,
        dim=cfg.model.embedding_dim,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        neg_samples=cfg.model.negative_samples,
        epochs=epochs,
        seed=cfg.model.seed,
    )

@app.command("build-index")
def build_index(dataset: str = typer.Option(...)):
    cfg = load_config(dataset)
    build_index_run(
        artifacts_dir=cfg.paths.artifacts_dir,
        nlist=cfg.retrieval.faiss_nlist,
        nprobe=cfg.retrieval.faiss_nprobe,
    )

@app.command("train-reranker")
def train_reranker(dataset: str = typer.Option(...)):
    cfg = load_config(dataset)
    params = {
        "num_leaves": cfg.reranker.num_leaves,
        "learning_rate": cfg.reranker.learning_rate,
        "n_estimators": cfg.reranker.n_estimators,
        "min_data_in_leaf": cfg.reranker.min_data_in_leaf,
    }
    train_reranker_run(
        processed_dir=cfg.paths.processed_dir,
        artifacts_dir=cfg.paths.artifacts_dir,
        top_candidates=cfg.retrieval.top_candidates,
        nprobe=cfg.retrieval.faiss_nprobe,
        reranker_params=params,
    )

@app.command("eval-offline")
def eval_offline(
    dataset: str = typer.Option(...),
    k: int = typer.Option(10),
    save_md: str = typer.Option(None, help="Path to save a markdown results file"),
):
    cfg = load_config(dataset)
    eval_run(
        processed_dir=cfg.paths.processed_dir,
        artifacts_dir=cfg.paths.artifacts_dir,
        top_candidates=cfg.retrieval.top_candidates,
        nprobe=cfg.retrieval.faiss_nprobe,
        k=k,
        save_md=save_md,
    )

if __name__ == "__main__":
    app()
