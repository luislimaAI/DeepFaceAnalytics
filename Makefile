.PHONY: install-dev lint typecheck test benchmark benchmark-compare profile check

install-dev:
	python -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

typecheck:
	mypy deepface_analytics/

test:
	pytest --cov=deepface_analytics --cov-report=html --cov-fail-under=70

benchmark:
	pytest tests/test_benchmarks.py --benchmark-save=baseline

benchmark-compare:
	pytest tests/test_benchmarks.py --benchmark-compare=baseline

profile:
	python tasks/profile_run.py

check: lint typecheck test
