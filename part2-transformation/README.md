# Part 2 : Transformation (Sessions + Attribution)

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r code/requirements.txt
```

## Run (recommended): from raw exports -> Part 1 cleaning -> Part 2 outputs
```bash
python code/pipeline.py --raw-dir <raw_csv_folder> --run-dq-cleaning --output-dir ./output
```

## Run: from cleaned exports only
```bash
python code/pipeline.py --clean-dir <cleaned_csv_folder> --output-dir ./output
```

Outputs are written into `output/`.

## Channel inference (important for review)

This dataset's `utm_*` values appear hashed/opaque, so channel classification is intentionally conservative:
- **paid_search**: presence of click-ids like `gclid/gbraid/wbraid/msclkid`
- **paid_social**: presence of `fbclid/ttclid/igshid`
- **organic_search / organic_social**: based on referrer domains
- **referral**: non-Puffy external referrers
- **direct**: no referrer and no campaign indicators
- **unknown_paid**: campaign-tagged traffic we can’t confidently classify

The derived field is `sessions.landing_channel`, and attribution outputs use the 7-day lookback across sessions.
