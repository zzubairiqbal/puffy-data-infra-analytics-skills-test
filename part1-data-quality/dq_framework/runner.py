from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd

from .checks import run_all_checks, load_contract
from .types import CheckResult
from .utils import coalesce_identity, parse_json_safe

def _json_safe(obj):
    # Convert numpy/pandas scalars to Python scalars for JSON
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k:_json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def results_to_dict(results: list[CheckResult]) -> list[dict]:
    out=[]
    for r in results:
        d = {
            "check_id": r.check_id,
            "description": r.description,
            "severity": r.severity,
            "passed": r.passed,
            "details": _json_safe(r.details),
            "sample_path": r.sample_path
        }
        out.append(d)
    return out

def _mad(x: pd.Series) -> float:
    med = float(x.median())
    return float((x - med).abs().median())

def _robust_z(x: float, series: pd.Series) -> float:
    med = float(series.median())
    mad = _mad(series)
    if mad == 0:
        return 0.0
    return 0.6745 * (x - med) / mad

def _score_partition(results: list[CheckResult]) -> int:
    # Simple DQ score out of 100 based on failed check severities.
    weights = {"P0_BLOCKER": 30, "P1_HIGH": 15, "P2_MEDIUM": 5, "P3_INFO": 0}
    penalty = sum(weights.get(r.severity, 5) for r in results if not r.passed)
    return max(0, 100 - int(penalty))

def run_checks_on_folder(input_dir: str, output_dir: str) -> dict:
    input_dir=str(input_dir)
    output_dir=str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    violations_dir = Path(output_dir)/"violations"
    violations_dir.mkdir(exist_ok=True)

    contract = load_contract()

    # --- First pass: compute per-partition profiling metrics & schema snapshots ---
    partitions=[]
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(input_dir, fname)
        df = load_csv(fpath)
        file_date = fname.split("_")[1].split(".")[0]
        cols = list(df.columns)

        cid = coalesce_identity(df) if ("client_id" in cols or "clientId" in cols) else pd.Series([pd.NA]*len(df))
        identity_null_rate = float(cid.isna().mean()) if len(df) else 1.0

        pur = df[df.get("event_name")=="checkout_completed"].copy() if "event_name" in cols else df.iloc[0:0].copy()
        revenue_sum = 0.0
        purchase_count = int(len(pur))
        if purchase_count:
            j = pur["event_data"].apply(parse_json_safe)
            rev = pd.to_numeric(j.apply(lambda x: (x or {}).get("revenue")), errors="coerce")
            revenue_sum = float(rev.fillna(0).sum())

        partitions.append({
            "file": fname,
            "date": file_date,
            "rows": int(len(df)),
            "columns": cols,
            "identity_null_rate": identity_null_rate,
            "purchase_count": purchase_count,
            "revenue_sum": revenue_sum
        })

    prof_df = pd.DataFrame(partitions).sort_values("date")
    prof_df.to_csv(Path(output_dir)/"profile.csv", index=False)

    # --- Second pass: run per-partition checks ---
    all_results=[]
    file_summaries=[]
    for p in partitions:
        fname=p["file"]
        file_date=p["date"]
        df=load_csv(os.path.join(input_dir, fname))
        res = run_all_checks(df, fname, file_date, contract=contract)
        score = _score_partition(res)
        decision = "QUARANTINE" if any((not r.passed) and r.severity=="P0_BLOCKER" for r in res) else "WARN" if any(not r.passed for r in res) else "PASS"

        all_results.append({"file": fname, "date": file_date, "dq_score": score, "decision": decision, "results": results_to_dict(res)})

        file_summaries.append({
            "file": fname,
            "date": file_date,
            "dq_score": score,
            "decision": decision,
            "failed": sum(1 for r in res if not r.passed),
            "p0_failed": sum(1 for r in res if (not r.passed) and r.severity=="P0_BLOCKER"),
            "p1_failed": sum(1 for r in res if (not r.passed) and r.severity=="P1_HIGH"),
            "p2_failed": sum(1 for r in res if (not r.passed) and r.severity=="P2_MEDIUM")
        })

    # --- Cross-partition drift/anomaly checks ---
    cross_checks=[]
    if not prof_df.empty:
        # identity null rate anomaly (robust)
        for _, row in prof_df.iterrows():
            z = _robust_z(float(row["identity_null_rate"]), prof_df["identity_null_rate"])
            passed = abs(z) <= 3.5  # robust threshold
            severity = "P1_HIGH" if abs(z) > 4.5 else "P2_MEDIUM" if abs(z) > 3.5 else "P3_INFO"
            cross_checks.append({
                "date": row["date"],
                "check_id": "drift.identity_null_rate_robust_z",
                "severity": severity,
                "passed": bool(passed),
                "details": {"robust_z": float(z), "identity_null_rate": float(row["identity_null_rate"])}
            })

        # schema drift vs previous day (missing critical columns)
        critical = set(contract.get("required_base_columns", []))
        for i in range(1, len(prof_df)):
            prev_cols=set(prof_df.iloc[i-1]["columns"])
            cols=set(prof_df.iloc[i]["columns"])
            removed = sorted(list(prev_cols - cols))
            added = sorted(list(cols - prev_cols))
            # Missing critical columns is a blocker.
            missing_critical = sorted(list(critical - cols))
            passed = len(missing_critical)==0
            cross_checks.append({
                "date": prof_df.iloc[i]["date"],
                "check_id": "drift.schema_removed_or_missing_critical",
                "severity": "P0_BLOCKER" if not passed else ("P1_HIGH" if removed else "P3_INFO"),
                "passed": bool(passed),
                "details": {"removed_columns": removed, "added_columns": added, "missing_critical": missing_critical, "prev_date": prof_df.iloc[i-1]["date"]}
            })

    # write dq_score.csv
    pd.DataFrame(file_summaries)[["date","file","dq_score","decision","failed","p0_failed","p1_failed","p2_failed"]].to_csv(Path(output_dir)/"dq_score.csv", index=False)

    summary = {
        "contract_version": "expected_contract.json",
        "files": all_results,
        "file_summaries": file_summaries,
        "cross_partition_checks": cross_checks
    }
    with open(Path(output_dir)/"results.json","w") as f:
        json.dump(summary, f, indent=2, default=str)

    # markdown report
    lines=[]
    lines.append("# Data Quality Report\n\n")
    lines.append("## Summary\n\n")
    total_files=len(file_summaries)
    total_p0=sum(x["p0_failed"] for x in file_summaries)
    total_failed=sum(x["failed"] for x in file_summaries)
    avg_score = round(sum(x["dq_score"] for x in file_summaries)/total_files, 1) if total_files else 0
    lines.append(f"- Partitions checked: **{total_files}**\n")
    lines.append(f"- Total failed checks: **{total_failed}**\n")
    lines.append(f"- P0 blockers: **{total_p0}**\n")
    lines.append(f"- Avg DQ score: **{avg_score}/100**\n")

    lines.append("\n## Per-partition\n\n")
    for s in file_summaries:
        lines.append(f"- {s['date']} ({s['file']}): decision={s['decision']} score={s['dq_score']}, failed={s['failed']} (P0={s['p0_failed']}, P1={s['p1_failed']}, P2={s['p2_failed']})\n")

    if cross_checks:
        lines.append("\n## Cross-partition drift / anomaly signals\n\n")
        # show only failed
        fails=[c for c in cross_checks if not c["passed"]]
        if not fails:
            lines.append("- No cross-partition anomalies detected.\n")
        else:
            for c in fails:
                lines.append(f"- {c['date']}: {c['check_id']} [{c['severity']}] details={c['details']}\n")

    with open(Path(output_dir)/"report.md","w") as f:
        f.write("".join(lines))
    return summary
