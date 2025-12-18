"""
Lightweight self-test suite (no pytest dependency) to validate core logic.

Run:
  python part2-transformation/tests/self_test.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from pipeline import ensure_canonical_identity, sessionize, extract_purchases, attribute_purchases  # noqa

def _ts(dtobj):
    return dtobj.isoformat().replace("+00:00","Z")

def test_session_split_on_inactivity():
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"client_id":"u1","clientId":None,"page_url":"/?utm_source=google&utm_medium=cpc","referrer":None,"timestamp":_ts(base),"event_name":"page_viewed","event_data":"{}","user_agent":"UA"},
        {"client_id":"u1","clientId":None,"page_url":"/p/1","referrer":"https://google.com","timestamp":_ts(base+timedelta(minutes=10)),"event_name":"product_added_to_cart","event_data":"{}","user_agent":"UA"},
        # 31 minutes later -> new session
        {"client_id":"u1","clientId":None,"page_url":"/checkout","referrer":"/p/1","timestamp":_ts(base+timedelta(minutes=41)),"event_name":"checkout_started","event_data":"{}","user_agent":"UA"},
    ]
    df = pd.DataFrame(rows)
    df = ensure_canonical_identity(df)
    sessions, ev = sessionize(df, session_gap_minutes=30)
    assert sessions["session_id"].nunique() == 2, f"expected 2 sessions, got {sessions['session_id'].nunique()}"

def test_attribution_7_day_lookback():
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        # session 1: facebook paid
        {"client_id":"u1","clientId":None,"page_url":"/?utm_source=facebook&utm_medium=paid_social","referrer":None,"timestamp":_ts(base),"event_name":"page_viewed","event_data":"{}","user_agent":"UA"},
        {"client_id":"u1","clientId":None,"page_url":"/p/1","referrer":"https://facebook.com","timestamp":_ts(base+timedelta(minutes=1)),"event_name":"product_added_to_cart","event_data":"{}","user_agent":"UA"},
        # purchase 6 days later, direct
        {"client_id":"u1","clientId":None,"page_url":"/checkout","referrer":None,"timestamp":_ts(base+timedelta(days=6)),"event_name":"checkout_completed","event_data":'{"transaction_id":"t1","revenue":100,"items":[{"item_id":"x","quantity":1}]}',"user_agent":"UA"},
    ]
    df = pd.DataFrame(rows)
    df = ensure_canonical_identity(df)
    sessions, ev = sessionize(df, session_gap_minutes=30)
    purchases = extract_purchases(ev)
    attrib = attribute_purchases(purchases, sessions, lookback_days=7)
    assert len(attrib) == 1
    assert "first_click_channel" in attrib.columns
    # should not be empty/unknown
    assert str(attrib.iloc[0]["first_click_channel"]) not in ("", "unknown", "nan")

def main():
    test_session_split_on_inactivity()
    test_attribution_7_day_lookback()
    print("All self-tests passed.")

if __name__ == "__main__":
    main()
