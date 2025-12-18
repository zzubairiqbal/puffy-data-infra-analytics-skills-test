## Part 4 — Monitoring & Alerting Plan

### Goal
Ensure the marketing analytics pipeline is production-safe by detecting real issues quickly **without alert fatigue**. Monitoring is split into two layers:
1) **Data Quality Gate (Part 1)** — catches schema drift, tracking regressions, and purchase integrity problems at ingestion time.
2) **Analytics Health Monitoring (Part 4)** — watches business-critical metrics after transformations to detect silent failures and data drift.

---

### What we monitor (and why)

#### A) Ingestion / raw-data health (from Part 1)
These are the most common causes of “numbers look wrong” incidents:
- **Schema drift**: missing/renamed columns (e.g., identity or referrer)
- **Identity coverage**: sudden spikes in missing client identifiers
- **Purchase integrity**: revenue <= 0 purchases, empty item payloads, transaction_id collisions/replays

Actioning:
- **P0 blockers** quarantine partitions and prevent promotion to production tables.
- **P1 highs** load with explicit “at risk” flag + alert.

#### B) Transformed analytics health (from Part 2 outputs)
We monitor daily aggregates that are both:
- stable enough to baseline,
- and directly tied to business decisions.

Metrics:
- sessions
- purchases
- revenue
- purchase conversion per session
- direct traffic share (proxy for loss of attribution signals)

Why these work:
- they detect pipeline failures, tracking regressions, and channel classification drift, even if raw data “looks fine”.

---

### Alerting logic (avoid noise)
For each day we compare metrics to a rolling baseline (prior 7 days where available):
- use a robust z-score (median + MAD) to be resilient to outliers
- require minimum absolute change for volume metrics (e.g., revenue change > $5k AND > 35%) to avoid paging on small swings

Severity:
- **P0_BLOCKER**: large revenue/purchase/session deviations that imply major breakage
- **P1_HIGH**: material drift (conversion shift, direct share shift, moderate volume anomaly)

---

### Outputs
Running the monitoring job produces:
- `daily_metrics.csv` — daily KPI table
- `alerts.json` — machine-readable alerts
- `report.md` — human-readable summary suitable for Slack/email

This can be scheduled daily (Airflow/Cron) after Part 2 completes.
