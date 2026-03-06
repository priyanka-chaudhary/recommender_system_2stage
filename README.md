# Two-Stage Recommender for Editorial Homepage Recommendations

This project implements a production-style two-stage recommender system for editorial homepage recommendations using the Microsoft MIND Small dataset.

The system is built in two stages:
1. **Candidate generation** with a two-tower embedding model in PyTorch.
2. **Candidate reranking** with a LightGBM LambdaRank model.

For fast retrieval, item embeddings are indexed with **FAISS**. Recommendations are exposed through a **FastAPI** endpoint.

### 🎯 Project Goal
The goal was to build a realistic recommender pipeline similar to modern homepage feed systems, where:
* A lightweight retrieval model first generates a shortlist of candidates.
* A second ranking model then reorders those candidates for better relevance.

### 📊 Dataset
I used **Microsoft MIND Small**, focusing on implicit feedback:
* User click behavior and article impressions.
* Held-out next-click style evaluation.

---

### 🏗️ Architecture

#### Stage 1: Candidate Generation
A two-tower model learns user and item embeddings. Candidates are retrieved by nearest-neighbor search using **FAISS**.

#### Stage 2: Reranking
A **LightGBM LambdaRank** model reranks candidates using features like:
* Retrieval score from the two-tower model.
* Article popularity and 
* Candidate rank position
* Offline evaluation


---

### 📈 Results
The two-stage recommender outperformed the popularity baseline by **31.1% relative NDCG@10**. The recommender was evaluated against a popularity baseline. Popularity baseline means: recommend the same globally popular items to every user

> **Stored in:** `artifacts/mind/final_results_20k.md`

### ⚙️ Experiment Setup
* **Dataset:** Microsoft MIND Small
* **Max users:** 20,000
* **Two-tower model:** PyTorch
* **Retrieval:** FAISS exact inner-product search
* **Reranker:** LightGBM LambdaRank
* **Negative samples:** 50
* **Embedding dimension:** 64
* **Top candidates:** 500
* **Evaluation metrics:** `NDCG@10`, `Recall@10`, `MAP@10`


| Model | NDCG@10 | Recall@10 | MAP@10 |
| :--- | :--- | :--- | :--- |
| Popularity baseline | 0.02023 | 0.04895 | 0.01191 |
| **Two-stage recommender** | **0.02652** | **0.05776** | **0.01726** |
| **Relative lift** | **+31.1%** | **+18.0%** | **+45.0%** |


---

### 🛠️ Tech Stack
* **Languages & Data:** Python, Pandas
* **ML Frameworks:** PyTorch, LightGBM
* **Vector Search:** FAISS
* **Deployment:** FastAPI, Docker, Airflow template

---

### What I learned
This project helped me understand several core recommender system concepts:
* Why strong baselines matter
* Why candidate retrieval quality is critical
* Why offline ranking metrics can move differently from recall metrics
* How train/evaluation feature mismatch can degrade performance
* How negative sampling affects embedding quality

A key debugging insight was that the initial reranker underperformed because training and evaluation features were inconsistent. After aligning features, switching retrieval to exact search for the smaller item set, and increasing negative sampling, the model achieved a strong positive lift over baseline.

---

### 🔮 Next Steps
* Add **user-category affinity** features.
* Incorporate **freshness and recency** features.
* Validate the pipeline on **RetailRocket** for ecommerce.

---

### 🚀 API Example
**Endpoint:** `GET /recommend?dataset=mind&user_id=U10022&k=10`

```json
{
  "dataset": "mind",
  "user_id": "U10022",
  "k": 10,
  "recommendations": [
    {"item_id": "N26331", "score": 1.8667},
    {"item_id": "N23446", "score": 1.5115}
  ]
}
```

----

This repo is a runnable template for a two stage recommender:
1) Candidate generation with a two tower model plus FAISS ANN retrieval  
2) A LightGBM reranker that reorders top candidates for the final top K

It supports two public datasets:
- MIND (editorial or news recommendations)
- RetailRocket (e commerce clickstream)

---

## 0) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put raw data here:
- data/raw/mind/news.tsv and data/raw/mind/behaviors.tsv
- data/raw/retailrocket/events.csv

---

## 1) Download datasets

MIND: https://msnews.github.io/  
RetailRocket: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

---

## 2) Run the pipeline (local, no Airflow)

MIND:
```bash
python -m src.cli prepare-data --dataset mind --max-users 20000
python -m src.cli train-two-tower --dataset mind --epochs 3
python -m src.cli build-index --dataset mind
python -m src.cli train-reranker --dataset mind
python -m src.cli eval-offline --dataset mind --k 10
```

Running experiments
```
EXP=mind_50k_min5_e10_neg50
EXP_DIR=artifacts/mind/experiments/$EXP
mkdir -p "$EXP_DIR"

# snapshot config before run
cp configs/mind.yaml "$EXP_DIR/mind.yaml"

# run pipeline
python -m src.cli prepare-data --dataset mind --max-users 50000

python -c "import pandas as pd; t=pd.read_parquet('data/processed/mind/train.parquet'); te=pd.read_parquet('data/processed/mind/test.parquet'); print('users', t.user_id.nunique(), 'items', t.item_id.nunique(), 'train', len(t), 'test', len(te))" | tee "$EXP_DIR/data_stats.txt"

python -m src.cli train-two-tower --dataset mind --epochs 10
python -m src.cli build-index --dataset mind
python -m src.cli train-reranker --dataset mind
python -m src.cli eval-offline --dataset mind --k 10 --save-md "$EXP_DIR/results.md"

# append the exact config used into results.md for zero ambiguity
echo -e "\n## Config used\n\n\`\`\`yaml" >> "$EXP_DIR/results.md"
cat "$EXP_DIR/mind.yaml" >> "$EXP_DIR/results.md"
echo -e "\n\`\`\`\n" >> "$EXP_DIR/results.md"

echo "Saved results to $EXP_DIR/results.md"
```

RetailRocket:
```bash
python -m src.cli prepare-data --dataset retailrocket --max-users 200000
python -m src.cli train-two-tower --dataset retailrocket --epochs 3
python -m src.cli build-index --dataset retailrocket
python -m src.cli train-reranker --dataset retailrocket
python -m src.cli eval-offline --dataset retailrocket --k 10
```

---
## Experiment log

This section tracks experiments so results are reproducible and comparable.
Important: changing `min_user_interactions` changes the evaluation cohort. That can change the strength of the popularity baseline, so comparisons across different cohort filters should be done carefully.

All experiment outputs are saved under `artifacts/mind/experiments/<experiment_id>/` and include:
- `results.md` (metrics)
- `mind.yaml` (config snapshot)
- `data_stats.txt` (users, items, interaction counts)

### MIND Small experiments (max_users = 50k)

| Experiment ID | Cohort filter | Epochs | Neg samples | Users | Items | Train interactions | Test interactions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mind_50k_min5_e10_neg50` | min_user_interactions = 5 | 10 | 50 | 16,159 | 6,700 | 149,083 | 16,159 |
| `mind_50k_min10_e20_neg100` | min_user_interactions = 10 | 20 | 100 | 5,836 | 5,670 | 92,693 | 5,836 |

### Metrics summary (k = 10)

| Experiment ID | Baseline NDCG@10 | Two-stage NDCG@10 | Lift | Baseline Recall@10 | Two-stage Recall@10 | Baseline MAP@10 | Two-stage MAP@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mind_50k_min5_e10_neg50` | 0.01429 | 0.02745 | +92.1% | 0.03558 | 0.05972 | 0.00821 | 0.01785 |
| `mind_50k_min10_e20_neg100` | 0.00694 | 0.03071 | +342.6% | 0.01885 | 0.06631 | 0.00358 | 0.02013 |

Notes:
- The large relative lifts at higher `min_user_interactions` happen partly because the baseline becomes weaker on a “high-activity user” cohort. Absolute metrics are often more stable for comparison.
- Full outputs and configs:
  - `artifacts/mind/experiments/mind_50k_min5_e10_neg50/results.md`
  - `artifacts/mind/experiments/mind_50k_min10_e20_neg100/results.md`
---

## Baselines

Current:
1) Popularity baseline  
Recommends the globally most popular items from the training set to every user.

Planned additions:
A) Two-tower retrieval baseline (no reranker)  
Ranks candidates purely by the two-tower / FAISS retrieval score. This isolates the incremental value of the LightGBM reranker.

B) Category-popularity baseline (light personalization)  
For each user, infer their preferred category from training clicks, then recommend the most popular items within that category. This is a stronger baseline than global popularity, especially for high-activity users.

## 3) Serve recommendations

```bash
uvicorn src.serving.app:app --reload --port 8000
curl "http://127.0.0.1:8000/recommend?dataset=mind&user_id=123&k=10"
```

---
