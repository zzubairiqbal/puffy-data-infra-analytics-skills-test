PY?=python
RAW_DIR?=./data/raw
CLEAN_DIR?=./data/cleaned
QUAR_DIR?=./data/quarantine

.PHONY: setup part1 part2 part3 part4 all test

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r part1-data-quality/requirements.txt && pip install -r part2-transformation/code/requirements.txt && pip install -r part3-analysis/code/requirements.txt && pip install -r part4-monitoring/code/requirements.txt

part1:
	$(PY) part1-data-quality/run_checks.py --input-dir $(RAW_DIR) --output-dir part1-data-quality/reports
	$(PY) part1-data-quality/export_cleaned.py --input-dir $(RAW_DIR) --clean-dir $(CLEAN_DIR) --quarantine-dir $(QUAR_DIR)

part2:
	$(PY) part2-transformation/code/pipeline.py --clean-dir $(CLEAN_DIR) --output-dir part2-transformation/output

part3:
	$(PY) part3-analysis/code/analyze.py --input-dir part2-transformation/output --output-dir part3-analysis

part4:
	$(PY) part4-monitoring/code/monitor.py --part2-output-dir part2-transformation/output --output-dir part4-monitoring/output --dq-results-json part1-data-quality/reports/results.json

all: part1 part2 part3 part4

test:
	$(PY) part2-transformation/tests/self_test.py
