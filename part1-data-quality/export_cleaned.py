from __future__ import annotations
import os, json, argparse
from pathlib import Path
import pandas as pd
from dq_framework.utils import coalesce_identity, parse_json_safe, stable_hash, ensure_columns

BASE_COLS = ["client_id","clientId","page_url","referrer","timestamp","event_name","event_data","user_agent"]

def clean_one_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # standardize columns
    df = ensure_columns(df.copy(), BASE_COLS)
    df["client_id_canonical"] = coalesce_identity(df)
    # ensure referrer col exists
    if "referrer" not in df.columns:
        df["referrer"] = pd.NA

    quarantine_rows = []
    keep_mask = pd.Series(True, index=df.index)

    # purchase-specific validation/quarantine
    pur_mask = df["event_name"]=="checkout_completed"
    pur = df[pur_mask].copy()
    if not pur.empty:
        j = pur["event_data"].apply(parse_json_safe)
        tx = j.apply(lambda x: (x or {}).get("transaction_id"))
        rev = pd.to_numeric(j.apply(lambda x: (x or {}).get("revenue")), errors="coerce").fillna(0)
        items = j.apply(lambda x: (x or {}).get("items"))
        bad = tx.isna() | (tx=="") | (rev<=0) | items.isna() | items.apply(lambda x: not isinstance(x, list) or len(x)==0)

        # quarantine invalid purchases
        bad_idx = pur[bad].index
        if len(bad_idx)>0:
            q = df.loc[bad_idx].copy()
            q["quarantine_reason"] = "invalid_purchase_fields_or_revenue"
            quarantine_rows.append(q)
            keep_mask.loc[bad_idx] = False

        # collisions / replays among remaining purchases
        pur_ok = df[pur_mask & keep_mask].copy()
        if not pur_ok.empty:
            j2 = pur_ok["event_data"].apply(parse_json_safe)
            pur_ok["tx"] = j2.apply(lambda x: (x or {}).get("transaction_id"))
            pur_ok["payload_hash"] = j2.apply(stable_hash)
            nunique = pur_ok.groupby("tx")["payload_hash"].nunique(dropna=False)
            collision_tx = nunique[nunique>1].index.tolist()
            if collision_tx:
                coll_idx = pur_ok[pur_ok["tx"].isin(collision_tx)].index
                q = df.loc[coll_idx].copy()
                q["quarantine_reason"] = "transaction_id_collision"
                quarantine_rows.append(q)
                keep_mask.loc[coll_idx] = False

            # exact replays: same tx + same payload duplicated -> keep earliest by timestamp
            pur_ok2 = df[pur_mask & keep_mask].copy()
            if not pur_ok2.empty:
                j3 = pur_ok2["event_data"].apply(parse_json_safe)
                pur_ok2["tx"] = j3.apply(lambda x: (x or {}).get("transaction_id"))
                pur_ok2["payload_hash"] = j3.apply(stable_hash)
                pur_ok2["ts"] = pd.to_datetime(pur_ok2["timestamp"], errors="coerce", utc=True)
                pur_ok2 = pur_ok2.sort_values("ts")
                dup = pur_ok2.duplicated(subset=["tx","payload_hash"], keep="first")
                dup_idx = pur_ok2[dup].index
                if len(dup_idx)>0:
                    q = df.loc[dup_idx].copy()
                    q["quarantine_reason"] = "duplicate_purchase_replay"
                    quarantine_rows.append(q)
                    keep_mask.loc[dup_idx] = False

    cleaned = df[keep_mask].copy()
    quarantined = pd.concat(quarantine_rows, ignore_index=True) if quarantine_rows else pd.DataFrame(columns=list(df.columns)+["quarantine_reason"])
    return cleaned, quarantined

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--quarantine-dir", required=True)
    args = ap.parse_args()
    Path(args.clean_dir).mkdir(parents=True, exist_ok=True)
    Path(args.quarantine_dir).mkdir(parents=True, exist_ok=True)

    summary=[]
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(args.input_dir, fname))
        cleaned, quarantined = clean_one_df(df)
        cleaned_out = Path(args.clean_dir)/fname.replace("events_","events_cleaned_")
        quarantined_out = Path(args.quarantine_dir)/fname.replace("events_","events_quarantine_")
        cleaned.to_csv(cleaned_out, index=False)
        quarantined.to_csv(quarantined_out, index=False)
        reason_counts = quarantined.get("quarantine_reason")
        reason_breakdown = reason_counts.value_counts().to_dict() if reason_counts is not None and len(quarantined) else {}
        summary.append({
            "file": fname,
            "clean_rows": int(len(cleaned)),
            "quarantine_rows": int(len(quarantined)),
            "quarantine_reason_counts": reason_breakdown
        })
    pd.DataFrame(summary).to_csv(Path(args.clean_dir).parent/"cleaning_summary.csv", index=False)
    # machine-readable manifest for downstream (Part 2/monitoring)
    manifest_path = Path(args.clean_dir).parent / "cleaning_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"partitions": summary}, f, indent=2)
    print("Wrote cleaned and quarantine exports.")

if __name__ == "__main__":
    main()
