# Part 1 : Data Quality Gate

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run validations
```bash
python run_checks.py --input-dir <raw_csv_folder> --output-dir ./reports
```

## Generate cleaned + quarantine exports (optional, used by Part 2)
```bash
python export_cleaned.py --input-dir <raw_csv_folder> --clean-dir ./cleaned --quarantine-dir ./quarantine
```
