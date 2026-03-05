# Two-Stage Recommender for Editorial Homepage Recommendations

This project implements a production-style two-stage recommender system for editorial homepage recommendations using the Microsoft MIND Small dataset.

The system is built in two stages:

Candidate generation with a two-tower embedding model in PyTorch

Candidate reranking with a LightGBM LambdaRank model

For fast retrieval, item embeddings are indexed with FAISS. Recommendations are exposed through a FastAPI endpoint.

### Project goal

The goal of the project was to build a realistic recommender pipeline similar to modern homepage feed systems, where:

a lightweight retrieval model first generates a shortlist of candidates

a second ranking model then reorders those candidates for better relevance

This setup reflects how large-scale recommender systems are often designed in practice, especially when ranking across large candidate spaces.

## Dataset

I used Microsoft MIND Small, a public dataset for news recommendation research.

The project focuses on implicit feedback:

user click behavior

article impressions

held-out next-click style evaluation

Architecture
Stage 1: Candidate generation

A two-tower model learns:

user embeddings

item embeddings

Candidate articles are retrieved by nearest-neighbor search over item embeddings using FAISS.

Stage 2: Reranking

A LightGBM LambdaRank model reranks retrieved candidates using features such as:

retrieval score from the two-tower model

article popularity

candidate rank position

Offline evaluation

The recommender was evaluated against a popularity baseline.

Popularity baseline means:

recommend the same globally popular items to every user

Main metric:

NDCG@10

Additional metrics:

Recall@10

MAP@10

Best result

On the strongest run, the two-stage recommender outperformed the popularity baseline by 31.1% relative NDCG@10.

Results
Model	NDCG@10	Recall@10	MAP@10
Popularity baseline	0.02023	0.04895	0.01191
Two-stage recommender	0.02652	0.05776	0.01726
Relative lift	+31.1%	+18.0%	+45.0%
What I learned

This project helped me understand several core recommender system concepts:

why strong baselines matter

why candidate retrieval quality is critical

why offline ranking metrics can move differently from recall metrics

how train/evaluation feature mismatch can degrade performance

how negative sampling affects embedding quality

A key debugging insight was that the initial reranker underperformed because training and evaluation features were inconsistent. After aligning features, switching retrieval to exact search for the smaller item set, and increasing negative sampling, the model achieved a strong positive lift over baseline.

API example

After training, recommendations can be served through FastAPI.

Example endpoint:

GET /recommend?dataset=mind&user_id=U10022&k=10

Example response:

{
  "dataset": "mind",
  "user_id": "U10022",
  "k": 10,
  "recommendations": [
    {"item_id": "N26331", "score": 1.8667},
    {"item_id": "N23446", "score": 1.5115}
  ]
}
Tech stack

Python

PyTorch

FAISS

LightGBM

Pandas

FastAPI

Docker

Airflow template

Next steps

Planned improvements:

add user-category affinity features

add freshness and recency features

run larger experiments on MIND

validate the same pipeline on RetailRocket for ecommerce recommendation

Save final metrics clearly

Create a simple text file so you always know which run belongs to which result.

Make a file like artifacts/mind/final_results_20k.md with this content:

Final Results - MIND Small
Experiment setup

Dataset: Microsoft MIND Small

Max users: 20000

Two-tower model: PyTorch

Retrieval: FAISS exact inner-product search

Reranker: LightGBM LambdaRank

Negative samples: 50

Embedding dimension: 64

Top candidates: 500

Evaluation metric: NDCG@10, Recall@10, MAP@10

Metrics
Model	NDCG@10	Recall@10	MAP@10
Popularity baseline	0.0202306399	0.0489477577	0.0119065762
Two-stage recommender	0.0265156036	0.0577556646	0.0172610643
Relative lift

NDCG@10: +31.1%

Recall@10: +18.0%

MAP@10: +45.0%


#?????????????????????????????????????????????????????????????????????????????????????????????????????????????

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

RetailRocket:
```bash
python -m src.cli prepare-data --dataset retailrocket --max-users 200000
python -m src.cli train-two-tower --dataset retailrocket --epochs 3
python -m src.cli build-index --dataset retailrocket
python -m src.cli train-reranker --dataset retailrocket
python -m src.cli eval-offline --dataset retailrocket --k 10
```

---

## 3) Serve recommendations

```bash
uvicorn src.serving.app:app --reload --port 8000
curl "http://127.0.0.1:8000/recommend?dataset=mind&user_id=123&k=10"
```

---

## CV bullet (fill measured values)

Built a production style two stage recommender for a homepage feed using Microsoft MIND and RetailRocket. Implemented two tower embedding retrieval with FAISS (top [200] candidates) and a LightGBM reranker, improving offline NDCG@10 by [+X%] vs a popularity baseline. Automated training and indexing via Airflow and Docker and served recommendations through a FastAPI endpoint.
