from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DATASET = "mind"

with DAG(
    dag_id="two_stage_recsys_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    prepare = BashOperator(
        task_id="prepare_data",
        bash_command=f"python -m src.cli prepare-data --dataset {DATASET}",
    )

    train_two_tower = BashOperator(
        task_id="train_two_tower",
        bash_command=f"python -m src.cli train-two-tower --dataset {DATASET} --epochs 3",
    )

    build_index = BashOperator(
        task_id="build_faiss_index",
        bash_command=f"python -m src.cli build-index --dataset {DATASET}",
    )

    train_reranker = BashOperator(
        task_id="train_reranker",
        bash_command=f"python -m src.cli train-reranker --dataset {DATASET}",
    )

    eval_offline = BashOperator(
        task_id="eval_offline",
        bash_command=f"python -m src.cli eval-offline --dataset {DATASET} --k 10",
    )

    prepare >> train_two_tower >> build_index >> train_reranker >> eval_offline
