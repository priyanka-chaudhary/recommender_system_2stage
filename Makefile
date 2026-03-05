mind:
	python -m src.cli prepare-data --dataset mind --max-users 20000
	python -m src.cli train-two-tower --dataset mind --epochs 3
	python -m src.cli build-index --dataset mind
	python -m src.cli train-reranker --dataset mind
	python -m src.cli eval-offline --dataset mind --k 10

retail:
	python -m src.cli prepare-data --dataset retailrocket --max-users 200000
	python -m src.cli train-two-tower --dataset retailrocket --epochs 3
	python -m src.cli build-index --dataset retailrocket
	python -m src.cli train-reranker --dataset retailrocket
	python -m src.cli eval-offline --dataset retailrocket --k 10

serve:
	uvicorn src.serving.app:app --reload --port 8000
