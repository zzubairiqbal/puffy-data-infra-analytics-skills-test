from __future__ import annotations
import hashlib, json
from urllib.parse import urlparse, parse_qs
import pandas as pd

def stable_hash(obj) -> str:
    """Stable hash for dict/list/string for collision detection."""
    if obj is None:
        s = ""
    elif isinstance(obj, (dict, list)):
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    else:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def parse_json_safe(x):
    if pd.isna(x) or x is None:
        return None
    try:
        return json.loads(x)
    except Exception:
        return None

def coalesce_identity(df: pd.DataFrame) -> pd.Series:
    a = df["client_id"] if "client_id" in df.columns else pd.Series([pd.NA]*len(df), index=df.index)
    b = df["clientId"] if "clientId" in df.columns else pd.Series([pd.NA]*len(df), index=df.index)
    return a.fillna(b)

def ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def parse_utm(url: str | None) -> dict:
    if not url or pd.isna(url):
        return {}
    try:
        qs = parse_qs(urlparse(url).query)
        # flatten singletons
        out = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k,v in qs.items()}
        return out
    except Exception:
        return {}
