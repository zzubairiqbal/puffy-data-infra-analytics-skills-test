from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def robust_z(x: float, med: float, mad: float) -> float:
    if mad == 0 or np.isnan(mad):
        return 0.0
    return 0.6745 * (x - med) / mad

def build_daily_metrics(part2_dir: Path) -> pd.DataFrame:
    sessions = pd.read_csv(part2_dir/"sessions.csv")
    purchases = pd.read_csv(part2_dir/"purchases.csv")

    sessions["day"] = pd.to_datetime(sessions["session_start_ts"], utc=True, format="mixed", errors="coerce").dt.date.astype(str)
    purchases["day"] = pd.to_datetime(purchases["purchase_ts"], utc=True, format="mixed", errors="coerce").dt.date.astype(str)

    daily = sessions.groupby("day")["session_id"].nunique().reset_index(name="sessions")
    p = purchases.groupby("day").agg(purchases=("transaction_id","nunique"),
                                     revenue=("revenue","sum")).reset_index()
    daily = daily.merge(p, on="day", how="left").fillna(0)
    daily["purchase_rate"] = np.where(daily["sessions"]>0, daily["purchases"]/daily["sessions"], np.nan)

    # channel mix (share of direct sessions)
    # Prefer the fact table if available (it has sessions by day/channel/device).
    if (part2_dir/"fact_marketing_performance_daily.csv").exists():
        fact = pd.read_csv(part2_dir/"fact_marketing_performance_daily.csv")
        fact["day"] = fact["day"].astype(str)
        direct = fact[fact["landing_channel"].fillna("unknown").astype(str).str.contains("direct", case=False, na=False)]
        mix = fact.groupby("day").apply(lambda g: pd.Series({
            "direct_session_share": float(direct[direct["day"]==g.name]["sessions"].sum() / max(g["sessions"].sum(), 1)),
            "direct_revenue_share": float(direct[direct["day"]==g.name]["revenue"].sum() / max(g["revenue"].sum(), 1)),
        })).reset_index()
        daily = daily.merge(mix, on="day", how="left")
    elif (part2_dir/"funnel_daily_by_channel.csv").exists():
        funnel = pd.read_csv(part2_dir/"funnel_daily_by_channel.csv")
        mix = funnel.groupby("event_day").apply(lambda g: pd.Series({
            "direct_session_share": float(g.loc[g["landing_channel"].fillna("unknown").astype(str).str.contains("direct")=="direct","sessions"].sum() / max(g["sessions"].sum(),1))
        })).reset_index().rename(columns={"event_day":"day"})
        daily = daily.merge(mix, on="day", how="left")
    return daily.sort_values("day")

def detect_alerts(daily: pd.DataFrame, lookback: int = 7) -> list[dict]:
    alerts=[]
    daily = daily.sort_values("day").reset_index(drop=True)
    metrics = ["sessions","purchases","revenue","purchase_rate","direct_session_share","direct_revenue_share"]
    for i in range(len(daily)):
        day = daily.loc[i,"day"]
        hist = daily.loc[max(0,i-lookback):i-1]
        if len(hist) < 3:
            continue
        for m in metrics:
            if m not in daily.columns: 
                continue
            x = daily.loc[i,m]
            h = hist[m].dropna()
            if len(h) < 3 or np.isnan(x):
                continue
            med = float(np.median(h))
            mad = float(np.median(np.abs(h - med)))
            z = robust_z(float(x), med, mad)
            # thresholds depend on metric
            if m in ["revenue","purchases","sessions"]:
                # require both relative and absolute changes to avoid noise
                abs_delta = float(x - med)
                rel = abs_delta / med if med != 0 else np.inf
                if (abs(rel) >= 0.35 and abs_delta >= (50 if m!="revenue" else 5000)) or abs(z) >= 4:
                    sev = "P1_HIGH" if abs(rel) < 0.75 else "P0_BLOCKER"
                    alerts.append({
                        "day": day,
                        "metric": m,
                        "value": float(x),
                        "baseline_median": med,
                        "mad": mad,
                        "robust_z": float(z),
                        "relative_change": float(rel),
                        "severity": sev,
                        "message": f"{m} deviated materially vs prior {lookback} days (median={med:.2f})."
                    })
            elif m == "purchase_rate":
                abs_delta = float(x - med)
                if abs_delta >= 0.01 and abs(z) >= 3:
                    alerts.append({
                        "day": day,
                        "metric": m,
                        "value": float(x),
                        "baseline_median": med,
                        "mad": mad,
                        "robust_z": float(z),
                        "severity": "P1_HIGH",
                        "message": "Conversion rate shifted vs baseline; investigate tracking or UX changes."
                    })
            elif m == "direct_session_share":
                abs_delta = float(x - med)
                if abs_delta >= 0.15 and abs(z) >= 3:
                    alerts.append({
                        "day": day,
                        "metric": m,
                        "value": float(x),
                        "baseline_median": med,
                        "mad": mad,
                        "robust_z": float(z),
                        "severity": "P1_HIGH",
                        "message": "Direct traffic share shifted materially; could indicate attribution signal loss (e.g., missing referrer/UTMs)."
                    })
    return alerts

def write_report(out_dir: Path, daily: pd.DataFrame, alerts: list[dict], dq_summary: dict | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir/"daily_metrics.csv", index=False)
    with open(out_dir/"alerts.json","w") as f:
        json.dump(alerts, f, indent=2)

    lines=[]
    lines.append("# Production Monitoring Report\n\n")
    lines.append("## Daily metrics\n\n")
    lines.append(daily.tail(14).to_markdown(index=False))
    lines.append("\n\n## Alerts\n\n")
    if not alerts:
        lines.append("No alerts triggered.\n")
    else:
        for a in alerts:
            lines.append(f"- **{a['severity']}** {a['day']} — {a['metric']}: {a['value']:.4g} (baseline median {a['baseline_median']:.4g}, z={a['robust_z']:.2f})\n")
            lines.append(f"  - {a['message']}\n")
    if dq_summary:
        lines.append("\n## Data Quality Gate summary (Part 1)\n\n")
        summaries = dq_summary.get("file_summaries", [])
        lines.append(f"- Partitions checked: {len(summaries)}\n")
        p0 = sum(x.get("p0_failed", 0) for x in summaries)
        lines.append(f"- Total P0 blockers (partition-level): {p0}\n")
        blocked = [x for x in summaries if x.get("decision") == "QUARANTINE" or x.get("p0_failed",0) > 0]
        if blocked:
            lines.append("\n**Blocked / at-risk partitions (P0):**\n")
            for b in blocked:
                lines.append(f"- {b.get('date')} ({b.get('file')}): score={b.get('dq_score')}\n")
        lines.append("\n(When a partition is blocked, use quarantine exports to prevent revenue overcounting.)\n")
    (out_dir/"report.md").write_text("".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part2-output-dir", required=True, help="Folder containing Part 2 outputs (sessions.csv, purchases.csv, etc.)")
    ap.add_argument("--output-dir", required=True, help="Folder to write monitoring outputs")
    ap.add_argument("--dq-results-json", help="Optional: path to Part 1 reports/results.json")
    args = ap.parse_args()

    part2_dir = Path(args.part2_output_dir)
    daily = build_daily_metrics(part2_dir)
    alerts = detect_alerts(daily, lookback=7)

    dq_summary=None
    if args.dq_results_json:
        try:
            with open(args.dq_results_json,"r") as f:
                dq_summary = json.load(f)
        except Exception:
            dq_summary = None

    write_report(Path(args.output_dir), daily, alerts, dq_summary)
    print(f"Wrote monitoring outputs to {args.output_dir}")

if __name__ == "__main__":
    main()
