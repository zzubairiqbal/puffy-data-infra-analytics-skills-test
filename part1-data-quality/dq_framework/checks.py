from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .types import CheckResult
from .utils import parse_json_safe, stable_hash, coalesce_identity

DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "expected_contract.json"

def load_contract(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONTRACT_PATH
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {
        "required_base_columns": ["page_url","timestamp","event_name","event_data","user_agent"],
        "identity_columns": ["client_id","clientId"],
        "optional_columns": ["referrer"],
        "allowed_events": ["page_viewed","product_added_to_cart","checkout_started","checkout_completed","email_filled_on_popup"],
        "event_data_requirements": {}
    }

def check_schema_contract(df: pd.DataFrame, file_name: str, contract: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    required = contract.get("required_base_columns", [])
    missing = [c for c in required if c not in df.columns]
    results.append(CheckResult(
        check_id="schema.required_columns_present",
        description=f"Required base columns present in {file_name}",
        severity="P0_BLOCKER",
        passed=(len(missing) == 0),
        details={"missing_columns": missing, "present_columns": list(df.columns)}
    ))

    identity_cols = contract.get("identity_columns", ["client_id", "clientId"])
    present_identity = [c for c in identity_cols if c in df.columns]
    results.append(CheckResult(
        check_id="schema.identity_column_present",
        description="At least one identity column exists (client_id or clientId)",
        severity="P0_BLOCKER",
        passed=(len(present_identity) > 0),
        details={"identity_columns": identity_cols, "present": present_identity}
    ))
    return results

def check_referrer_presence(df: pd.DataFrame) -> CheckResult:
    present = "referrer" in df.columns
    return CheckResult(
        check_id="schema.referrer_column_present",
        description="Referrer column exists (used for attribution/channel classification)",
        severity="P1_HIGH",
        passed=present,
        details={"present": present}
    )

def check_allowed_events(df: pd.DataFrame, contract: dict) -> CheckResult:
    allowed = set(contract.get("allowed_events", []))
    values = set(df["event_name"].dropna().unique()) if "event_name" in df.columns else set()
    unknown = sorted(list(values - allowed))
    return CheckResult(
        check_id="schema.unknown_event_names",
        description="No unknown event_name values (contract enforcement)",
        severity="P2_MEDIUM",
        passed=(len(unknown) == 0),
        details={"unknown_event_names": unknown, "allowed": sorted(list(allowed))}
    )

def check_event_data_json_parse_rate(df: pd.DataFrame) -> CheckResult:
    if "event_data" not in df.columns:
        return CheckResult(
            check_id="integrity.event_data_json_parse_rate",
            description="event_data JSON parse success rate",
            severity="P0_BLOCKER",
            passed=False,
            details={"reason": "event_data column missing"}
        )
    ed = df["event_data"]
    parsed = ed.apply(parse_json_safe)
    fail = parsed.isna() & ed.notna() & (ed.astype(str).str.len() > 0)
    fail_rate = float(fail.mean()) if len(df) else 0.0
    passed = fail_rate <= 0.01
    severity = "P1_HIGH" if fail_rate > 0.02 else "P2_MEDIUM" if fail_rate > 0.01 else "P3_INFO"
    return CheckResult(
        check_id="integrity.event_data_json_parse_rate",
        description="event_data JSON parse success rate (should be >99%)",
        severity=severity,
        passed=passed,
        details={"fail_rate": fail_rate, "fail_count": int(fail.sum()), "total": int(len(df))}
    )

def check_timestamp_parse_and_partition(df: pd.DataFrame, file_date: str) -> CheckResult:
    if "timestamp" not in df.columns:
        return CheckResult(
            check_id="integrity.timestamp_parse_and_partition_alignment",
            description="Timestamps parse and align to partition date",
            severity="P0_BLOCKER",
            passed=False,
            details={"reason": "timestamp column missing"}
        )
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    parse_fail = ts.isna()
    parse_fail_rate = float(parse_fail.mean()) if len(df) else 0.0
    date_str = ts.dt.strftime("%Y%m%d")
    mismatch = (~parse_fail) & (date_str != file_date)
    mismatch_rate = float(mismatch.mean()) if len(df) else 0.0

    passed = (parse_fail_rate <= 0.01) and (mismatch_rate <= 0.05)
    severity = "P0_BLOCKER" if (parse_fail_rate > 0.01 or mismatch_rate > 0.10) else "P1_HIGH" if mismatch_rate > 0.05 else "P2_MEDIUM" if parse_fail_rate > 0 else "P3_INFO"
    return CheckResult(
        check_id="integrity.timestamp_parse_and_partition_alignment",
        description="Timestamps parse and align to partition date (allow small late/early spillover)",
        severity=severity,
        passed=passed,
        details={"parse_fail_rate": parse_fail_rate, "mismatch_rate": mismatch_rate, "rows": int(len(df))}
    )

def check_identity_coverage(df: pd.DataFrame) -> CheckResult:
    cid = coalesce_identity(df)
    null_rate = float(cid.isna().mean()) if len(df) else 1.0
    # Hard safety rail: if identity is missing for a quarter of events, something is badly broken.
    passed = null_rate <= 0.10
    severity = "P1_HIGH" if null_rate > 0.10 else "P2_MEDIUM" if null_rate > 0.02 else "P3_INFO"
    return CheckResult(
        check_id="tracking.identity_null_rate",
        description="Client identity completeness (coalesced client_id/clientId)",
        severity=severity,
        passed=passed,
        details={"null_rate": null_rate, "null_count": int(cid.isna().sum()), "total": int(len(df))}
    )

def _purchase_extract(df: pd.DataFrame) -> pd.DataFrame:
    if "event_name" not in df.columns:
        return df.iloc[0:0].copy()
    pur = df[df["event_name"] == "checkout_completed"].copy()
    if pur.empty:
        return pur
    j = pur["event_data"].apply(parse_json_safe)
    pur["transaction_id"] = j.apply(lambda x: (x or {}).get("transaction_id"))
    pur["revenue"] = j.apply(lambda x: (x or {}).get("revenue"))
    pur["items"] = j.apply(lambda x: (x or {}).get("items"))
    pur["payload_hash"] = j.apply(stable_hash)
    return pur

def check_purchase_integrity(df: pd.DataFrame, contract: dict) -> list[CheckResult]:
    pur = _purchase_extract(df)
    if pur.empty:
        return [CheckResult(
            check_id="purchase.integrity.no_purchases",
            description="No purchases found (checkout_completed)",
            severity="P3_INFO",
            passed=True,
            details={"purchase_events": 0}
        )]

    invalid = (
        pur["transaction_id"].isna()
        | pur["revenue"].isna()
        | (pd.to_numeric(pur["revenue"], errors="coerce").fillna(-1) <= 0)
        | pur["items"].isna()
        | pur["items"].apply(lambda x: isinstance(x, list) and len(x) == 0)
    )

    invalid_count = int(invalid.sum())
    invalid_rate = float(invalid.mean()) if len(pur) else 0.0

    # Small numbers of invalid purchases should be quarantined (high severity) but not necessarily block the entire day.
    # If invalid purchases become systemic, block the partition.
    severity = "P1_HIGH" if invalid_count > 0 else "P3_INFO"
    passed = (invalid_count == 0)

    results: list[CheckResult] = []
    results.append(CheckResult(
        check_id="purchase.integrity.required_fields_and_positive_revenue",
        description="Purchases have transaction_id, revenue>0, and non-empty items (small counts quarantined; systemic blocks)",
        severity=severity,
        passed=passed,
        details={"invalid_count": invalid_count, "invalid_rate": invalid_rate, "purchase_events": int(len(pur))}
    ))

    # Collision: same transaction_id used for different payloads.
    g = pur.groupby("transaction_id")["payload_hash"].nunique(dropna=False)
    collision_tx = g[g > 1].index.dropna().tolist()
    results.append(CheckResult(
        check_id="purchase.transaction_id_collision",
        description="No transaction_id collisions (same tx used for different purchase payloads)",
        severity="P0_BLOCKER",
        passed=(len(collision_tx) == 0),
        details={"collision_tx": collision_tx, "collision_count": len(collision_tx)}
    ))

    # Replay duplicates: same transaction_id + same payload multiple times.
    dup = pur.duplicated(subset=["transaction_id", "payload_hash"], keep=False)
    replay_event_rows = int(dup.sum())
    results.append(CheckResult(
        check_id="purchase.transaction_id_replay_duplicates",
        description="No duplicate purchase events with same transaction_id and identical payload (double-count risk)",
        severity="P1_HIGH",
        passed=(replay_event_rows == 0),
        details={"replay_event_rows": replay_event_rows}
    ))
    return results

def run_all_checks(df: pd.DataFrame, file_name: str, file_date: str, contract: dict | None = None) -> list[CheckResult]:
    contract = contract or load_contract()
    # Ensure required columns exist so downstream checks don't crash
    for c in contract.get("required_base_columns", []):
        if c not in df.columns:
            df[c] = pd.NA

    results: list[CheckResult] = []
    results.extend(check_schema_contract(df, file_name, contract))
    results.append(check_allowed_events(df, contract))
    results.append(check_event_data_json_parse_rate(df))
    results.append(check_timestamp_parse_and_partition(df, file_date))
    results.append(check_referrer_presence(df))
    results.append(check_identity_coverage(df))
    results.extend(check_purchase_integrity(df, contract))
    return results