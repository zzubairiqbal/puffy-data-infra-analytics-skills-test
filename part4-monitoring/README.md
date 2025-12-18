# Part 4 : Monitoring

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## Run
```bash
python code/monitor.py --part2-output-dir <part2_output_dir> --output-dir ./output
```

Optional (include Part 1 DQ summary):
```bash
python code/monitor.py --part2-output-dir <part2_output_dir> --output-dir ./output --dq-results-json <path_to_part1_reports_results.json>
```
