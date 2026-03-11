.PHONY: install test lint smoke-tiny train-300m

install:
	pip install -e ".[all]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check aletheia/ tests/

smoke-tiny:
	python -m aletheia.training.pretrain --config configs/model/bitlinear_tiny.yaml --max-steps 10

train-300m:
	python -m aletheia.training.pretrain --config configs/model/bitlinear_300m.yaml
