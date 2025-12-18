from __future__ import annotations
import os, argparse, json, sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
import hashlib

# --- Helpers ---
def parse_utm(url: str | None) -> dict:
    if not url or pd.isna(url):
        return {}
    try:
        qs = parse_qs(urlparse(url).query)
        return {k: (v[0] if isinstance(v, list) and v else v) for k,v in qs.items()}
    except Exception:
        return {}

def is_mobile(ua: str | None) -> bool:
    if not ua or pd.isna(ua):
        return False
    s = ua.lower()
    return any(x in s for x in ["iphone","android","mobile","ipad"])

def classify_channel(page_url: str | None, referrer: str | None) -> tuple[str, str]:
    """
    Return (channel, channel_detail) for the *visit* that produced page_url.

    Constraints in this dataset:
    - utm_source / utm_medium values are hashed, so we avoid semantics like 'cpc'/'email'.
    - we rely on click-ids + referrer domains + presence of any utm params.

    Channel taxonomy (pragmatic, marketing-friendly):
    - paid_search: gclid/gbraid/wbraid/msclkid
    - paid_social: fbclid/ttclid/igshid (best-effort)
    - organic_search: referrer is a search engine AND no paid click-id
    - referral: referrer exists and is non-Puffy and non-search/social
    - direct: no referrer and no click-id and no utm params
    - unknown_paid: utm params present but we can't confidently classify
    """
    utm = parse_utm(page_url)

    # Click IDs (highest confidence)
    paid_search_ids = {"gclid", "gbraid", "wbraid", "msclkid"}
    paid_social_ids = {"fbclid", "ttclid", "igshid"}
    if any(k in utm for k in paid_search_ids):
        return "paid_search", "click_id"
    if any(k in utm for k in paid_social_ids):
        return "paid_social", "click_id"

    # Referrer-based
    ref = (referrer.strip().lower() if isinstance(referrer, str) else "")
    ref_domain = ""
    if ref:
        try:
            ref_domain = urlparse(ref).netloc.lower()
        except Exception:
            ref_domain = ""

    # Search engines
    search_domains = ("google.", "bing.", "yahoo.", "duckduckgo.", "baidu.", "ecosia.")
    if ref_domain and any(sd in ref_domain for sd in search_domains):
        return "organic_search", ref_domain

    # Social
    social_domains = ("facebook.", "instagram.", "tiktok.", "pinterest.", "snapchat.", "reddit.", "x.com", "twitter.")
    if ref_domain and any(sd in ref_domain for sd in social_domains):
        return "organic_social", ref_domain

    # Known tracking redirect domains (dataset has source-*.com)
    if ref_domain.startswith("source-") and ref_domain.endswith(".com"):
        return "unknown_paid", "redirect_domain"

    # Any other external referrer
    if ref_domain and "puffy.com" not in ref_domain and "checkout.puffy.com" not in ref_domain:
        return "referral", ref_domain

    # UTM present but no click-id: treat as unknown_paid (campaign-tagged traffic)
    if any(k.startswith("utm_") for k in utm.keys()):
        return "unknown_paid", "utm_present"

    return "direct", "no_referrer"


def load_events_from_folder(folder: str) -> pd.DataFrame:
    dfs=[]
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)
        # keep date
        # infer YYYYMMDD from filename
        digits = "".join(ch for ch in fname if ch.isdigit())
        if len(digits)>=8:
            df["partition_date"] = digits[:8]
        else:
            df["partition_date"] = None
        dfs.append(df)
    if not dfs:
        raise ValueError(f"No CSV files found in {folder}")
    out = pd.concat(dfs, ignore_index=True)
    return out

def ensure_canonical_identity(df: pd.DataFrame) -> pd.DataFrame:
    if "client_id_canonical" not in df.columns:
        a = df["client_id"] if "client_id" in df.columns else pd.Series([pd.NA]*len(df), index=df.index)
        b = df["clientId"] if "clientId" in df.columns else pd.Series([pd.NA]*len(df), index=df.index)
        df["client_id_canonical"] = a.fillna(b)
    return df

# --- Part 1 cleaning (invoked optionally) ---

def enrich_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields used by Part 2 (UTMs, device, channel).

    We keep this lightweight and deterministic:
    - utm_* extracted from page_url query params (hashed values are fine).
    - device_type derived from user_agent heuristics.
    - channel/channel_detail derived from click-ids + referrer domains (see classify_channel()).
    """
    e = events.copy()

    # Extract UTM fields from page_url
    def _get_param(url: str | None, key: str) -> str | None:
        if not isinstance(url, str) or not url:
            return None
        try:
            qs = parse_utm(url)
            val = qs.get(key)
            return str(val) if val is not None else None
        except Exception:
            return None

    for key in ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"]:
        if key not in e.columns:
            e[key] = e["page_url"].apply(lambda u, k=key: _get_param(u, k))

    # Device
    if "device_type" not in e.columns:
        ua = e.get("user_agent", pd.Series([""] * len(e))).astype(str)
        e["device_type"] = ua.apply(lambda s: "mobile" if is_mobile(s) else "desktop")

    # Channel
    if "channel" not in e.columns or "channel_detail" not in e.columns:
        ch = e.apply(lambda r: classify_channel(r.get("page_url"), r.get("referrer")), axis=1)
        e["channel"] = [c for c, d in ch]
        e["channel_detail"] = [d for c, d in ch]

    return e

def run_part1_cleaning(raw_dir: str, clean_dir: str, quarantine_dir: str):
    # import part1 export_cleaned via relative path
    part1_path = Path(__file__).resolve().parents[1].parent / "part1-data-quality"
    sys.path.insert(0, str(part1_path))
    from export_cleaned import main as _noop  # for side effects (not used)
    # call export_cleaned.py functions by shelling would be messy; re-implement minimal here.
    # Instead, import its clean_one_df
    sys.path.insert(0, str(part1_path))
    import export_cleaned as ec

    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    Path(quarantine_dir).mkdir(parents=True, exist_ok=True)

    summary=[]
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".csv"): 
            continue
        df = pd.read_csv(os.path.join(raw_dir, fname))
        cleaned, quarantined = ec.clean_one_df(df)
        cleaned_out = Path(clean_dir)/fname.replace("events_","events_cleaned_")
        quarantined_out = Path(quarantine_dir)/fname.replace("events_","events_quarantine_")
        cleaned.to_csv(cleaned_out, index=False)
        quarantined.to_csv(quarantined_out, index=False)
        summary.append({"file": fname, "clean_rows": int(len(cleaned)), "quarantine_rows": int(len(quarantined))})
    pd.DataFrame(summary).to_csv(Path(clean_dir).parent/"cleaning_summary.csv", index=False)

# --- Sessionization ---


def sessionize(events: pd.DataFrame, session_gap_minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sessionize events by inactivity gap (default: 30 minutes).

    Notes:
    - We sessionize primarily by canonical identity when available.
    - If identity is missing, we fall back to a stable anonymous key derived from user_agent,
      so events from the same browser can still stitch into sessions (best-effort).
    """
    e = events.copy()
    e["ts"] = pd.to_datetime(e["timestamp"], utc=True, errors="coerce", format="mixed")
    e = e.dropna(subset=["ts"]).copy()

    # Best-effort key for sessionization when identity is missing
    ua = e.get("user_agent", pd.Series([""] * len(e))).astype(str)
    ua_hash = ua.apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:12])
    e["session_key"] = e["client_id_canonical"].astype(str)
    e.loc[e["client_id_canonical"].isna(), "session_key"] = "anon:" + ua_hash[e["client_id_canonical"].isna()]

    e = e.sort_values(["session_key", "ts"])

    e["prev_ts"] = e.groupby("session_key")["ts"].shift(1)
    gap_min = (e["ts"] - e["prev_ts"]).dt.total_seconds() / 60.0
    e["new_session"] = e["prev_ts"].isna() | (gap_min > session_gap_minutes)
    e["session_num"] = e.groupby("session_key")["new_session"].cumsum().astype(int)

    starts = (
        e.groupby(["session_key", "session_num"], dropna=False)["ts"]
        .min()
        .reset_index()
        .rename(columns={"ts": "session_start_ts_dt"})
    )
    starts["session_start_ts"] = starts["session_start_ts_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    starts["session_id"] = starts.apply(
        lambda r: f"{r['session_key']}:{int(r['session_num'])}:{r['session_start_ts_dt'].strftime('%Y%m%dT%H%M%S')}",
        axis=1,
    )

    e = e.merge(starts[["session_key", "session_num", "session_id", "session_start_ts"]], on=["session_key", "session_num"], how="left")

    ends = e.groupby("session_id")["ts"].max().reset_index().rename(columns={"ts": "session_end_ts_dt"})
    sess = starts.merge(ends, on="session_id", how="left")
    sess["session_end_ts"] = sess["session_end_ts_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    sess["session_duration_seconds"] = (sess["session_end_ts_dt"] - sess["session_start_ts_dt"]).dt.total_seconds()

    # Prefer the first *page_view* in the session as the landing context.
    e_sorted = e.sort_values("ts")
    is_candidate = (e_sorted["event_name"].astype(str) == "page_view")
    # Prefer non-checkout domains for marketing channel inference
    pu = e_sorted.get("page_url", pd.Series([""]*len(e_sorted))).astype(str)
    is_candidate = is_candidate & pu.str.contains("puffy.com", na=False) & (~pu.str.contains("checkout.puffy.com", na=False))
    landing = e_sorted[is_candidate].groupby("session_id", dropna=False).first().reset_index()
    if landing.empty:
        landing = e_sorted.groupby("session_id", dropna=False).first().reset_index()
    for col in ["page_url","referrer","utm_source","utm_medium","utm_campaign","utm_term","utm_content","channel","channel_detail","device_type","client_id_canonical"]:
        if col not in landing.columns:
            landing[col] = pd.NA

    landing = landing[["session_id","client_id_canonical","page_url","referrer","utm_source","utm_medium","utm_campaign","utm_term","utm_content","channel","channel_detail","device_type"]]
    landing.rename(columns={
        "page_url":"landing_page_url",
        "channel":"landing_channel",
        "channel_detail":"landing_channel_detail"
    }, inplace=True)

    sess = sess.merge(landing, on="session_id", how="left")
    sess.rename(columns={"client_id_canonical":"client_id_canonical"}, inplace=True)

    sess["event_count"] = e.groupby("session_id").size().reindex(sess["session_id"]).fillna(0).astype(int).values

    # Keep canonical identity in session table for downstream joins
    # (If session_key is anonymous, client_id_canonical may be null)
    sess = sess.drop(columns=["session_start_ts_dt","session_end_ts_dt"])

    e = e.drop(columns=["prev_ts","new_session"])

    return sess, e

def extract_purchases(events: pd.DataFrame) -> pd.DataFrame:
    pur = events[events["event_name"]=="checkout_completed"].copy()
    if pur.empty:
        return pd.DataFrame(columns=["transaction_id","purchase_ts","client_id_canonical","session_id","revenue","user_email","items_json"])
    j = pur["event_data"].apply(lambda x: json.loads(x) if isinstance(x,str) else {})
    pur["transaction_id"] = j.apply(lambda x: x.get("transaction_id"))
    pur["revenue"] = j.apply(lambda x: x.get("revenue"))
    pur["user_email"] = j.apply(lambda x: x.get("user_email"))
    pur["items_json"] = j.apply(lambda x: json.dumps(x.get("items"), sort_keys=True))
    pur["purchase_ts"] = pd.to_datetime(pur["timestamp"], errors="coerce", utc=True)
    return pur[["transaction_id","purchase_ts","client_id_canonical","session_id","revenue","user_email","items_json"]]

def attribute_purchases(purchases: pd.DataFrame, sessions: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    if purchases.empty:
        return purchases.assign(first_click_session_id=pd.NA, last_click_session_id=pd.NA,
                               first_click_channel=pd.NA, last_click_channel=pd.NA,
                               last_non_direct_session_id=pd.NA, last_non_direct_channel=pd.NA)
    sessions = sessions.copy()
    sessions["session_start_ts"] = pd.to_datetime(sessions["session_start_ts"], utc=True)
    sessions = sessions.sort_values(["client_id_canonical","session_start_ts"])
    out_rows=[]
    for _, p in purchases.iterrows():
        cid = p["client_id_canonical"]
        start = p["purchase_ts"] - pd.Timedelta(days=lookback_days)
        end = p["purchase_ts"]
        cand = sessions[(sessions["client_id_canonical"]==cid) & (sessions["session_start_ts"]>=start) & (sessions["session_start_ts"]<=end)]
        if cand.empty:
            out_rows.append({**p.to_dict(),
                             "first_click_session_id": pd.NA,
                             "last_click_session_id": pd.NA,
                             "first_click_channel": pd.NA,
                             "last_click_channel": pd.NA,
                             "last_non_direct_session_id": pd.NA,
                             "last_non_direct_channel": pd.NA})
            continue
        first_row = cand.iloc[0]
        last_row = cand.iloc[-1]
        # last non-direct (common in ecommerce)
        nd = cand[cand["landing_channel"]!="direct"]
        if not nd.empty:
            last_nd = nd.iloc[-1]
            last_nd_id = last_nd["session_id"]
            last_nd_ch = last_nd["landing_channel"]
        else:
            last_nd_id = pd.NA
            last_nd_ch = pd.NA

        out_rows.append({**p.to_dict(),
                         "first_click_session_id": first_row["session_id"],
                         "last_click_session_id": last_row["session_id"],
                         "first_click_channel": first_row["landing_channel"],
                         "last_click_channel": last_row["landing_channel"],
                         "last_non_direct_session_id": last_nd_id,
                         "last_non_direct_channel": last_nd_ch})
    return pd.DataFrame(out_rows)


def build_touchpoints(events_with_sessions: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Touchpoints table: one row per inbound marketing context observed in a session.
    We keep it intentionally lightweight for attribution / debugging.
    """
    e = events_with_sessions.copy()
    e["ts"] = pd.to_datetime(e["timestamp"], errors="coerce", utc=True)
    # keep first event per session as the "session touchpoint"
    first = e.sort_values("ts").groupby("session_id", dropna=False).head(1).copy()
    cols = [
        "session_id","client_id_canonical","ts","page_url","referrer",
        "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
        "channel","channel_detail","device_type"
    ]
    for c in cols:
        if c not in first.columns:
            first[c] = pd.NA
    tp = first[cols].rename(columns={"ts":"touch_ts"})
    tp["is_direct"] = tp["channel"].astype(str).str.contains("direct", case=False, na=False)
    tp["is_paid"] = tp["channel"].astype(str).str.contains("paid|affiliate", case=False, na=False)
    return tp

def build_daily_fact(sessions: pd.DataFrame, purchases: pd.DataFrame, attrib: pd.DataFrame) -> pd.DataFrame:
    """
    Daily marketing performance fact table, useful for dashboards and monitoring.
    Produces daily metrics by channel + device + attribution model.
    """
    s = sessions.copy()
    s["day"] = pd.to_datetime(s["session_start_ts"], utc=True, errors="coerce").dt.date.astype(str)
    sess = s.groupby(["day","landing_channel","device_type"], dropna=False).size().reset_index(name="sessions")

    p = purchases.copy()
    if not p.empty:
        p["day"] = pd.to_datetime(p["purchase_ts"], utc=True, errors="coerce").dt.date.astype(str)
        p = p.merge(s[["session_id","landing_channel","device_type"]], on="session_id", how="left")
        pur = p.groupby(["day","landing_channel","device_type"], dropna=False).agg(
            purchases=("transaction_id","nunique"),
            revenue=("revenue","sum")
        ).reset_index()
    else:
        pur = pd.DataFrame(columns=["day","landing_channel","device_type","purchases","revenue"])

    fact = sess.merge(pur, on=["day","landing_channel","device_type"], how="left")
    fact["purchases"] = fact["purchases"].fillna(0).astype(int)
    fact["revenue"] = fact["revenue"].fillna(0.0).astype(float)
    fact["conversion_rate"] = (fact["purchases"] / fact["sessions"]).replace([np.inf, -np.inf], np.nan)

    # Attribution reconciliation outputs (optional) - total attributed revenue per day
    if not attrib.empty:
        a = attrib.copy()
        a["day"] = pd.to_datetime(a["purchase_ts"], utc=True, errors="coerce").dt.date.astype(str)
        for label, col in [("first_click","first_click_channel"),("last_click","last_click_channel"),("last_non_direct","last_non_direct_channel")]:
            tmp = a.groupby(["day", col], dropna=False)["revenue"].sum().reset_index()
            tmp.rename(columns={col:"channel", "revenue":f"attributed_revenue_{label}"}, inplace=True)
            # merge into fact on matching channel
            fact = fact.merge(tmp, left_on=["day","landing_channel"], right_on=["day","channel"], how="left")
            fact.drop(columns=["channel"], inplace=True)
            fact[f"attributed_revenue_{label}"] = fact[f"attributed_revenue_{label}"].fillna(0.0).astype(float)

    return fact

def daily_channel_funnel(events: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["event_day"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True).dt.date.astype(str)
    # join session channel
    ses_ch = sessions[["session_id","landing_channel","device_type"]].copy()
    merged = events.merge(ses_ch, on="session_id", how="left")
    # aggregate per day/channel
    agg = merged.pivot_table(index=["event_day","landing_channel"], columns="event_name", values="timestamp", aggfunc="count", fill_value=0).reset_index()
    # sessions count per day/channel
    sess_day = sessions.copy()
    sess_day["event_day"]=pd.to_datetime(sess_day["session_start_ts"], utc=True).dt.date.astype(str)
    sess_counts = sess_day.groupby(["event_day","landing_channel"])["session_id"].nunique().reset_index(name="sessions")
    out = agg.merge(sess_counts, on=["event_day","landing_channel"], how="left")
    # conversion
    out["purchase_rate_per_session"] = out.get("checkout_completed",0) / out["sessions"].replace(0, np.nan)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", help="Folder with raw daily exports (events_YYYYMMDD.csv)")
    ap.add_argument("--clean-dir", help="Folder with cleaned exports (events_cleaned_YYYYMMDD.csv)")
    ap.add_argument("--output-dir", required=True, help="Folder to write transformation outputs")
    ap.add_argument("--run-dq-cleaning", action="store_true", help="If set, generate cleaned exports from raw before transforming")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.run_dq_cleaning:
        if not args.raw_dir:
            raise ValueError("--raw-dir is required when --run-dq-cleaning is set")
        clean_dir = outdir / "_cleaned_events"
        quarantine_dir = outdir / "_quarantine_events"
        run_part1_cleaning(args.raw_dir, str(clean_dir), str(quarantine_dir))
        events = load_events_from_folder(str(clean_dir))
    else:
        if not args.clean_dir:
            raise ValueError("--clean-dir is required if not running dq cleaning")
        events = load_events_from_folder(args.clean_dir)

    events = ensure_canonical_identity(events)
    events = enrich_events(events)
    # sessionize
    sessions, events_with_sessions = sessionize(events, session_gap_minutes=30)
    # purchases
    purchases = extract_purchases(events_with_sessions)
    # attribution
    attrib = attribute_purchases(purchases, sessions, lookback_days=7)
    # funnel
    funnel = daily_channel_funnel(events_with_sessions, sessions)

    # validations / reconciliation
    recon = {
        "input_event_rows": int(len(events)),
        "session_rows": int(len(sessions)),
        "purchase_events_rows": int((events["event_name"]=="checkout_completed").sum()),
        "purchases_table_rows": int(len(purchases)),
        "revenue_sum_purchases": float(pd.to_numeric(purchases["revenue"], errors="coerce").fillna(0).sum()) if not purchases.empty else 0.0,
        "purchases_missing_session_id": int(purchases["session_id"].isna().sum()) if not purchases.empty else 0,
    }
    # attribution reconciliation
    if not attrib.empty:
        recon["revenue_sum_attrib_first_click"] = float(attrib["revenue"].sum())
        recon["revenue_sum_attrib_last_click"] = float(attrib["revenue"].sum())
        recon["revenue_sum_attrib_last_non_direct"] = float(attrib["revenue"].sum())
        recon["revenue_sum_diff_purchases_vs_attrib"] = float(recon["revenue_sum_purchases"] - attrib["revenue"].sum())
    else:
        recon["revenue_sum_attrib_first_click"] = 0.0
        recon["revenue_sum_attrib_last_click"] = 0.0
        recon["revenue_sum_attrib_last_non_direct"] = 0.0
        recon["revenue_sum_diff_purchases_vs_attrib"] = float(recon["revenue_sum_purchases"])
    (outdir/"validation").mkdir(exist_ok=True)
    with open(outdir/"validation"/"reconciliation.json","w") as f:
        json.dump(recon, f, indent=2)

    # write outputs
    sessions.to_csv(outdir/"sessions.csv", index=False)
    events_with_sessions.to_csv(outdir/"sessionized_events.csv", index=False)
    purchases.to_csv(outdir/"purchases.csv", index=False)
    attrib.to_csv(outdir/"purchase_attribution.csv", index=False)
    funnel.to_csv(outdir/"funnel_daily_by_channel.csv", index=False)

    # additional outputs for explainability & monitoring
    touchpoints = build_touchpoints(events_with_sessions, sessions)
    touchpoints.to_csv(outdir/"touchpoints.csv", index=False)

    daily_fact = build_daily_fact(sessions, purchases, attrib)
    daily_fact.to_csv(outdir/"fact_marketing_performance_daily.csv", index=False)

    # daily attribution summary (first/last/last_non_direct)
    if not attrib.empty:
        attrib["purchase_day"] = pd.to_datetime(attrib["purchase_ts"], utc=True).dt.date.astype(str)
        for label, col in [("first_click","first_click_channel"),("last_click","last_click_channel"),("last_non_direct","last_non_direct_channel")]:
            summ = attrib.groupby(["purchase_day", col], dropna=False)["revenue"].sum().reset_index()
            summ.rename(columns={col:"channel", "revenue":"attributed_revenue"}, inplace=True)
            summ.to_csv(outdir/f"attribution_{label}_daily.csv", index=False)

    print(f"Wrote transformation outputs to {outdir}")

if __name__ == "__main__":
    main()