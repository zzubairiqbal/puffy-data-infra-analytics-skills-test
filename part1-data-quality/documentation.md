## Part 1 : Data Quality Framework (incoming event exports)

### What we check (and why it matters for Puffy)
The goal is to catch tracking and revenue issues *before* they show up in dashboards or decision-making. I treated this like a production gate with two actions:
- **Block / quarantine a partition** when the data is structurally unsafe (schema breaks, transaction collisions).
- **Quarantine bad rows + alert** when the issue is row-level (a few malformed purchase payloads), so the rest of the day can still load.

Checks are grouped into three buckets:

1) **Contract / schema**
- Required columns exist (`page_url`, `timestamp`, `event_name`, `event_data`, `user_agent`)
- At least one identity column exists (`client_id` or `clientId`)
- Referrer column exists (critical for attribution and “direct vs paid” channel mix)

2) **Tracking integrity**
- Timestamps parse and mostly align with the partition date (guards against late/early spillover and parsing failures)
- Identity completeness: % of events missing a canonical client id (drives sessionization + attribution accuracy)

3) **Purchase integrity**
- Purchase payload has `transaction_id`, `revenue > 0`, and non-empty `items`
- **Transaction collisions**: same `transaction_id` appears with different payloads (highest risk for revenue overcount)
- Replay duplicates: exact duplicate purchase events (double-count risk)

### Severity and scoring (aligned to Puffy’s evaluation style)
- **P0_BLOCKER** → Block / quarantine the partition (downstream tables are “unsafe” for that day)
- **P1_HIGH** → Load with alert + row quarantine (downstream is usable but “at risk”)
- **P2_MEDIUM / P3_INFO** → Log quality debt / informational only

I also compute a simple **DQ score (0–100)** per day (weighted penalties by severity) so quality can trend over time.

### Exact issues found in this dataset
From the 14 daily files:

**1) Missing referrer column (attribution risk)**
- Referrer is missing for **5 consecutive partitions: 2025‑03‑04 → 2025‑03‑08**
- Impact: channel classification will drift toward “direct/unknown” and will distort marketing performance comparisons.

**2) Identity regression (sessionization + attribution accuracy)**
- Canonical identity null rate exceeded 10% on:
  - **2025‑03‑02 (~15.8% missing)**
  - **2025‑03‑08 (~14.3% missing)**
- Impact: more “anonymous” sessions, weaker deduping, and noisier conversion funnels by channel/device.

**3) Purchase payload quality (row quarantine)**
- Days with malformed purchase rows (missing/invalid revenue or items):
  - **2025‑02‑24, 02‑25, 02‑27, 02‑28, 2025‑03‑03, 03‑04, 03‑06**
  - Typically **1–2 bad purchase rows/day** (in some cases ~5–10% of that day’s purchases)
- Impact: if not filtered, these can cause revenue under/over-reporting and broken item-level analysis.

**4) Transaction ID collisions (P0 blockers — revenue overcount risk)**
- Detected collisions on:
  - **2025‑02‑26** (example tx: `ORD-20250226-400`)
  - **2025‑02‑27** (example tx: `ORD-20250227-262`)
  - **2025‑03‑01** (example tx: `ORD-20250301-176`)
  - **2025‑03‑08** (example tx: `ORD-20250308-149`)
- Impact: this is the cleanest explanation for “revenue is wrong” mid-window. If the same order id can represent different payloads, summing revenue becomes unreliable without upstream fixes.

### Outputs produced (so Part 2+ can trust the inputs)
- `part1-data-quality/reports/report.md` + `results.json` + `dq_score.csv` (human + machine readable)
- `data/cleaned/` (canonicalized columns + quarantined rows removed)
- `data/quarantine/` (bad rows with `quarantine_reason`)
- `cleaning_manifest.json` (row quarantine breakdown by day)

This gives a clean, reproducible handoff: **Part 2 reads from `data/cleaned/`** to ensure attribution + revenue metrics are based on safe events.
