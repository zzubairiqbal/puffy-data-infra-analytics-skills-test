from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader


def safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = pd.read_csv(input_dir / "sessions.csv")
    purchases = pd.read_csv(input_dir / "purchases.csv")
    attrib = pd.read_csv(input_dir / "purchase_attribution.csv")
    return sessions, purchases, attrib


def compute_metrics(sessions: pd.DataFrame, purchases: pd.DataFrame, attrib: pd.DataFrame) -> dict:
    sessions = sessions.copy()
    purchases = purchases.copy()
    attrib = attrib.copy()

    sessions["session_start_dt"] = safe_dt(sessions["session_start_ts"])
    purchases["purchase_dt"] = safe_dt(purchases["purchase_ts"])
    purchases["revenue"] = pd.to_numeric(purchases["revenue"], errors="coerce").fillna(0.0)
    attrib["revenue"] = pd.to_numeric(attrib["revenue"], errors="coerce").fillna(0.0)

    total_revenue = float(purchases["revenue"].sum())
    total_purchases = int(purchases["transaction_id"].nunique())
    total_sessions = int(sessions["session_id"].nunique())
    conv = (total_purchases / total_sessions) if total_sessions else float("nan")
    aov = (total_revenue / total_purchases) if total_purchases else float("nan")
    rps = (total_revenue / total_sessions) if total_sessions else float("nan")

    # daily
    sessions["day"] = sessions["session_start_dt"].dt.date.astype(str)
    purchases["day"] = purchases["purchase_dt"].dt.date.astype(str)

    daily_pur = purchases.groupby("day").agg(purchases=("transaction_id", "nunique"), revenue=("revenue", "sum")).reset_index()
    daily_sess = sessions.groupby("day").agg(sessions=("session_id", "nunique")).reset_index()
    daily = daily_sess.merge(daily_pur, on="day", how="left").fillna({"purchases": 0, "revenue": 0})
    daily["purchase_rate"] = daily["purchases"] / daily["sessions"].replace(0, np.nan)
    daily = daily.sort_values("day")

    # attribution (direct bucket for null last_non_direct)
    attrib["last_non_direct_channel"] = attrib["last_non_direct_channel"].fillna("direct")
    attrib["first_click_channel"] = attrib["first_click_channel"].fillna("direct")
    attrib["last_click_channel"] = attrib["last_click_channel"].fillna("direct")

    channel_rev = (
        attrib.groupby("last_non_direct_channel")["revenue"].sum().sort_values(ascending=False).reset_index()
        .rename(columns={"last_non_direct_channel": "channel"})
    )

    def rev_share(col: str) -> pd.Series:
        s = attrib.groupby(col)["revenue"].sum()
        return (s / s.sum()).sort_values(ascending=False)

    share_first = rev_share("first_click_channel")
    share_last = rev_share("last_click_channel")
    share_lastnd = rev_share("last_non_direct_channel")

    top_channels = list(dict.fromkeys(list(share_lastnd.index[:6]) + list(share_first.index[:6]) + list(share_last.index[:6])))
    attrib_tbl = pd.DataFrame({
        "channel": top_channels,
        "first_click": [float(share_first.get(c, 0.0)) for c in top_channels],
        "last_click": [float(share_last.get(c, 0.0)) for c in top_channels],
        "last_non_direct": [float(share_lastnd.get(c, 0.0)) for c in top_channels],
    })

    # device conversion
    sess_purch = sessions[["session_id", "device_type", "client_id_canonical", "session_start_dt"]].merge(
        purchases[["session_id"]].drop_duplicates(), on="session_id", how="left", indicator=True
    )
    sess_purch["has_purchase"] = sess_purch["_merge"] == "both"
    device = sess_purch.groupby("device_type").agg(sessions=("session_id", "nunique"), purchases=("has_purchase", "sum")).reset_index()
    device["purchase_rate"] = device["purchases"] / device["sessions"].replace(0, np.nan)

    # new vs returning (within window)
    sessions_sorted = sessions.sort_values(["client_id_canonical", "session_start_dt"])
    sessions_sorted["session_index"] = sessions_sorted.groupby("client_id_canonical").cumcount() + 1
    sessions_sorted["user_type"] = np.where(sessions_sorted["session_index"] == 1, "new", "returning")
    uv = sessions_sorted[["session_id", "user_type"]].merge(
        purchases[["session_id"]].drop_duplicates(), on="session_id", how="left", indicator=True
    )
    uv["has_purchase"] = uv["_merge"] == "both"
    user_conv = uv.groupby("user_type").agg(sessions=("session_id", "nunique"), purchases=("has_purchase", "sum")).reset_index()
    user_conv["purchase_rate"] = user_conv["purchases"] / user_conv["sessions"].replace(0, np.nan)

    # time-to-purchase
    ttp = purchases.merge(sessions[["session_id", "session_start_dt"]], on="session_id", how="left")
    ttp["minutes_to_purchase"] = (ttp["purchase_dt"] - ttp["session_start_dt"]).dt.total_seconds() / 60.0
    ttp = ttp.replace([np.inf, -np.inf], np.nan).dropna(subset=["minutes_to_purchase"])

    return {
        "sessions": sessions,
        "purchases": purchases,
        "attrib": attrib,
        "totals": {
            "total_revenue": total_revenue,
            "total_purchases": total_purchases,
            "total_sessions": total_sessions,
            "conversion": conv,
            "aov": aov,
            "rps": rps,
        },
        "daily": daily,
        "channel_rev": channel_rev,
        "attrib_tbl": attrib_tbl,
        "device": device,
        "user_conv": user_conv,
        "ttp": ttp,
        "shares": {
            "first": share_first,
            "last": share_last,
            "last_non_direct": share_lastnd,
        }
    }


def make_charts(out_dir: Path, m: dict, dq_scores: pd.DataFrame | None) -> dict[str, Path]:
    ensure_dir(out_dir)
    charts: dict[str, Path] = {}

    daily = m["daily"]

    # daily revenue
    p = out_dir / "daily_revenue.png"
    plt.figure(figsize=(7.2, 3))
    plt.plot(daily["day"], daily["revenue"], marker="o")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("Revenue ($)")
    plt.title("Daily revenue")
    save_fig(p)
    charts["daily_revenue"] = p

    # daily conversion
    p = out_dir / "daily_conversion.png"
    plt.figure(figsize=(7.2, 3))
    plt.plot(daily["day"], daily["purchase_rate"], marker="o")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("Purchases / sessions")
    plt.title("Daily session conversion")
    save_fig(p)
    charts["daily_conversion"] = p

    # channel revenue (last non-direct)
    ch = m["channel_rev"].copy()
    p = out_dir / "channel_revenue.png"
    plt.figure(figsize=(7.2, 3.2))
    plt.bar(ch["channel"].astype(str), ch["revenue"])
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.ylabel("Attributed revenue ($)")
    plt.title("Attributed revenue by channel (last non-direct, 7d lookback)")
    save_fig(p)
    charts["channel_revenue"] = p

    # device conversion
    dev = m["device"].copy()
    p = out_dir / "device_conversion.png"
    plt.figure(figsize=(3.5, 3))
    plt.bar(dev["device_type"].astype(str), dev["purchase_rate"])
    plt.ylabel("Purchases / sessions")
    plt.title("Conversion by device")
    save_fig(p)
    charts["device_conversion"] = p

    # new vs returning
    uv = m["user_conv"].copy()
    p = out_dir / "new_vs_returning_conversion.png"
    plt.figure(figsize=(3.5, 3))
    plt.bar(uv["user_type"].astype(str), uv["purchase_rate"])
    plt.ylabel("Purchases / sessions")
    plt.title("Conversion: new vs returning")
    save_fig(p)
    charts["new_vs_returning_conversion"] = p

    # time-to-purchase
    ttp = m["ttp"]
    p = out_dir / "time_to_purchase_hist.png"
    plt.figure(figsize=(3.5, 3))
    plt.hist(ttp["minutes_to_purchase"], bins=20)
    plt.xlabel("Minutes from session start")
    plt.ylabel("Purchases")
    plt.title("Time to purchase")
    save_fig(p)
    charts["time_to_purchase_hist"] = p

    # dq score by day (optional)
    if dq_scores is not None and not dq_scores.empty:
        p = out_dir / "dq_score.png"
        plt.figure(figsize=(7.2, 2.8))
        plt.plot(dq_scores["day"], dq_scores["dq_score"], marker="o")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylim(0, 105)
        plt.ylabel("DQ score (0-100)")
        plt.title("Data quality score by day")
        save_fig(p)
        charts["dq_score"] = p

    return charts


def draw_kpi_cards(c: canvas.Canvas, kpis: list[tuple[str, str]], y_top: float) -> None:
    W, _ = letter
    card_w = (W - 1.5 * inch - 0.5 * inch) / 3
    card_h = 0.85 * inch
    x0 = 0.75 * inch

    for i, (label, value) in enumerate(kpis):
        row = i // 3
        col = i % 3
        x = x0 + col * (card_w + 0.25 * inch)
        y = y_top - row * (card_h + 0.28 * inch)

        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.roundRect(x, y, card_w, card_h, 10, stroke=1, fill=0)
        c.setFont("Helvetica", 10)
        c.drawString(x + 0.18 * inch, y + card_h - 0.32 * inch, label)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x + 0.18 * inch, y + 0.22 * inch, value)


def build_pdf(out_pdf: Path, charts: dict[str, Path], m: dict, dq_summary: dict | None) -> None:
    W, H = letter
    c = canvas.Canvas(str(out_pdf), pagesize=letter)

    totals = m["totals"]
    shares = m["shares"]["last_non_direct"]

    # derive a few items for bullets
    device = m["device"].copy()
    user_conv = m["user_conv"].copy()
    ttp = m["ttp"]["minutes_to_purchase"]
    median_mins = float(ttp.median())
    p90_mins = float(ttp.quantile(0.9))

    desktop_rate = float(device.loc[device["device_type"] == "desktop", "purchase_rate"].iloc[0]) if (device["device_type"] == "desktop").any() else float("nan")
    mobile_rate = float(device.loc[device["device_type"] == "mobile", "purchase_rate"].iloc[0]) if (device["device_type"] == "mobile").any() else float("nan")
    mobile_share = float(device.loc[device["device_type"] == "mobile", "sessions"].iloc[0] / totals["total_sessions"]) if (device["device_type"] == "mobile").any() and totals["total_sessions"] else float("nan")

    new_rate = float(user_conv.loc[user_conv["user_type"] == "new", "purchase_rate"].iloc[0]) if (user_conv["user_type"] == "new").any() else float("nan")
    ret_rate = float(user_conv.loc[user_conv["user_type"] == "returning", "purchase_rate"].iloc[0]) if (user_conv["user_type"] == "returning").any() else float("nan")

    # header helper
    def header(title: str, subtitle: str) -> None:
        c.setFont("Helvetica-Bold", 20)
        c.drawString(0.75 * inch, H - 0.9 * inch, title)
        c.setFont("Helvetica", 10.5)
        c.setFillColor(colors.grey)
        c.drawString(0.75 * inch, H - 1.15 * inch, subtitle)
        c.setFillColor(colors.black)

    def footer(page_no: int) -> None:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawRightString(W - 0.75 * inch, 0.55 * inch, f"Page {page_no}  |  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        c.setFillColor(colors.black)

    # Page 1
    header(
        "Puffy Marketing Analytics - Executive Summary",
        "14-day window (2025-02-23 to 2025-03-08) | Sessions: 30-min inactivity | Attribution: 7-day lookback",
    )

    kpis = [
        ("Total revenue", money(totals["total_revenue"])),
        ("Purchases", f"{totals['total_purchases']:,}"),
        ("Sessions", f"{totals['total_sessions']:,}"),
        ("Session conversion", pct(totals["conversion"])),
        ("Avg order value (AOV)", money(totals["aov"])),
        ("Revenue / session", money(totals["rps"])),
    ]
    draw_kpi_cards(c, kpis, y_top=H - 2.35 * inch)

    # Takeaways
    direct_share = float(shares.get("direct", 0.0))
    paid_search_share = float(shares.get("paid_search", 0.0))
    unknown_paid_share = float(shares.get("unknown_paid", 0.0))
    organic_share = float(shares.get("organic_search", 0.0))

    bullets = [
        f"Revenue in this window is {money(totals['total_revenue'])} across {totals['total_purchases']} purchases (AOV {money(totals['aov'])}).",
        f"Session conversion is {pct(totals['conversion'])}. Returning sessions convert ~{(ret_rate / new_rate if new_rate else 0):.1f}x vs new ({pct(ret_rate)} vs {pct(new_rate)}).",
        f"Channel mix (last non-direct): direct {pct(direct_share)}, paid_search {pct(paid_search_share)}, unknown_paid {pct(unknown_paid_share)}, organic_search {pct(organic_share)}.",
        f"Desktop converts better than mobile ({pct(desktop_rate)} vs {pct(mobile_rate)}); mobile is ~{pct(mobile_share)} of sessions.",
        f"Purchase timing is fast: median {median_mins:.1f} min from session start; 90% within {p90_mins:.1f} min.",
    ]

    c.setFont("Helvetica-Bold", 13)
    c.drawString(0.75 * inch, H - 4.55 * inch, "Key takeaways")
    c.setFont("Helvetica", 10.5)
    y = H - 4.80 * inch
    for b in bullets:
        c.drawString(0.90 * inch, y, "• " + b)
        y -= 0.22 * inch

    # charts (two wide)
    chart_y = 1.85 * inch
    chart_h = 2.45 * inch
    chart_w = (W - 1.5 * inch - 0.25 * inch) / 2
    c.drawImage(ImageReader(str(charts["daily_revenue"])), 0.75 * inch, chart_y, width=chart_w, height=chart_h, preserveAspectRatio=True, mask="auto")
    c.drawImage(ImageReader(str(charts["daily_conversion"])), 0.75 * inch + chart_w + 0.25 * inch, chart_y, width=chart_w, height=chart_h, preserveAspectRatio=True, mask="auto")

    footer(1)
    c.showPage()

    # Page 2
    header(
        "Drivers: channel, device, and reliability",
        "Channel results are attribution-based (7-day lookback). Reliability notes reflect the raw data quality gate.",
    )

    c.drawImage(ImageReader(str(charts["channel_revenue"])), 0.75 * inch, H - 3.35 * inch, width=W - 1.5 * inch, height=2.0 * inch, preserveAspectRatio=True, mask="auto")

    # attribution table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, H - 3.6 * inch, "Attribution model sensitivity (revenue share)")
    attrib_tbl = m["attrib_tbl"].copy().head(6)
    table_data = [["Channel", "First-click", "Last-click", "Last non-direct"]]
    for _, r in attrib_tbl.iterrows():
        table_data.append([r["channel"], pct(r["first_click"]), pct(r["last_click"]), pct(r["last_non_direct"])])

    t = Table(table_data, colWidths=[2.0 * inch, 1.4 * inch, 1.4 * inch, 1.7 * inch])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    t_w, t_h = t.wrapOn(c, W, H)
    t_x = 0.75 * inch
    t_y = H - 4.15 * inch - t_h
    t.drawOn(c, t_x, t_y)

    # reliability band
    avg_dq = None
    p0_days = None
    if dq_summary:
        avg_dq = dq_summary.get("avg_dq")
        p0_days = dq_summary.get("p0_days")

    band_y = t_y - 0.65 * inch
    band_h = 0.62 * inch
    c.setStrokeColor(colors.black)
    c.roundRect(0.75 * inch, band_y, W - 1.5 * inch, band_h, 10, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.75 * inch + 0.18 * inch, band_y + band_h - 0.23 * inch, "Reliability notes (so we trust decisions)")
    c.setFont("Helvetica", 9.0)

    notes_left = [
        f"DQ score avg {avg_dq:.1f}/100; P0 issue days {p0_days}/14." if avg_dq is not None else "DQ report not provided; use results directionally.",
        "client_id field rename on 2025-02-27 (handled in cleaning).",
    ]
    notes_right = [
        "Referrer missing from 2025-03-04; channel after that is conservative.",
        "Txn id collisions / $0 purchases quarantined before analytics.",
    ]
    ny = band_y + 0.20 * inch
    c.drawString(0.75 * inch + 0.18 * inch, ny, "• " + notes_left[0])
    c.drawString(0.75 * inch + 0.18 * inch, ny - 0.17 * inch, "• " + notes_left[1])
    rx = 0.75 * inch + (W - 1.5 * inch) / 2
    c.drawString(rx + 0.1 * inch, ny, "• " + notes_right[0])
    c.drawString(rx + 0.1 * inch, ny - 0.17 * inch, "• " + notes_right[1])

    # chart grid
    grid_top = band_y - 0.25 * inch
    grid_bottom = 0.85 * inch
    grid_h = grid_top - grid_bottom
    cell_h = grid_h / 2 - 0.15 * inch
    cell_w = (W - 1.5 * inch - 0.25 * inch) / 2

    x_left = 0.75 * inch
    x_right = 0.75 * inch + cell_w + 0.25 * inch
    y_upper = grid_bottom + cell_h + 0.15 * inch
    y_lower = grid_bottom

    c.drawImage(ImageReader(str(charts["device_conversion"])), x_left, y_upper, width=cell_w, height=cell_h, preserveAspectRatio=True, mask="auto")
    c.drawImage(ImageReader(str(charts["new_vs_returning_conversion"])), x_right, y_upper, width=cell_w, height=cell_h, preserveAspectRatio=True, mask="auto")
    c.drawImage(ImageReader(str(charts["time_to_purchase_hist"])), x_left, y_lower, width=cell_w, height=cell_h, preserveAspectRatio=True, mask="auto")

    if "dq_score" in charts:
        c.drawImage(ImageReader(str(charts["dq_score"])), x_right, y_lower, width=cell_w, height=cell_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_right, y_lower + cell_h - 0.25 * inch, "Data quality score by day")
        c.setFont("Helvetica", 9.5)
        c.drawString(x_right, y_lower + cell_h - 0.50 * inch, "Provide Part 1 results.json to render this chart.")

    footer(2)
    c.save()


def load_dq_results(dq_results_path: Path | None) -> tuple[pd.DataFrame | None, dict | None]:
    if dq_results_path is None or not dq_results_path.exists():
        return None, None

    try:
        payload = json.loads(dq_results_path.read_text(encoding="utf-8"))
        fs = pd.DataFrame(payload.get("file_summaries", []))
        if fs.empty:
            return None, None
        fs["day"] = pd.to_datetime(fs["date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        fs = fs.dropna(subset=["day"])
        avg_dq = float(fs["dq_score"].mean())
        p0_days = int((fs["p0_failed"] > 0).sum())
        return fs[["day", "dq_score"]], {"avg_dq": avg_dq, "p0_days": p0_days}
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Path to Part 2 output folder (sessions.csv, purchases.csv, purchase_attribution.csv)")
    ap.add_argument("--output-dir", required=True, help="Path to Part 3 folder (will write PDF + supporting-analysis)")
    ap.add_argument("--dq-results", default=None, help="Optional path to Part 1 reports/results.json")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    charts_dir = out_dir / "supporting-analysis"
    ensure_dir(charts_dir)

    sessions, purchases, attrib = load_inputs(input_dir)
    m = compute_metrics(sessions, purchases, attrib)

    dq_scores, dq_summary = load_dq_results(Path(args.dq_results) if args.dq_results else None)

    charts = make_charts(charts_dir, m, dq_scores)

    # Save traceability tables
    m["daily"].to_csv(charts_dir / "daily_metrics.csv", index=False)
    m["channel_rev"].to_csv(charts_dir / "channel_revenue.csv", index=False)
    m["device"].to_csv(charts_dir / "device_conversion.csv", index=False)
    m["user_conv"].to_csv(charts_dir / "new_vs_returning_conversion.csv", index=False)
    m["attrib_tbl"].to_csv(charts_dir / "attrib_sensitivity.csv", index=False)
    pd.DataFrame([m["totals"]]).to_csv(charts_dir / "numbers_used.csv", index=False)

    # Build PDF
    out_pdf = out_dir / "executive-summary.pdf"
    build_pdf(out_pdf, charts, m, dq_summary)


if __name__ == "__main__":
    main()
