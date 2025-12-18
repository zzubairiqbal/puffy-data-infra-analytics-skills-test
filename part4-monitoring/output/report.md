# Production Monitoring Report

## Daily metrics

| day        |   sessions |   purchases |   revenue |   purchase_rate |   direct_session_share |   direct_revenue_share |
|:-----------|-----------:|------------:|----------:|----------------:|-----------------------:|-----------------------:|
| 2025-02-23 |       3393 |          21 |   20408   |      0.00618921 |                      0 |                      0 |
| 2025-02-24 |       3126 |          25 |   25457.6 |      0.00799744 |                      0 |                      0 |
| 2025-02-25 |       2853 |          15 |   13543   |      0.00525762 |                      0 |                      0 |
| 2025-02-26 |       2585 |          13 |   13906   |      0.00502901 |                      0 |                      0 |
| 2025-02-27 |       2821 |          18 |   14000   |      0.00638072 |                      0 |                      0 |
| 2025-02-28 |       2593 |          18 |   18669   |      0.00694177 |                      0 |                      0 |
| 2025-03-01 |       2818 |          27 |   29590   |      0.00958126 |                      0 |                      0 |
| 2025-03-02 |       3302 |          22 |   28221   |      0.00666263 |                      0 |                      0 |
| 2025-03-03 |       3195 |          22 |   20504.8 |      0.00688576 |                      0 |                      0 |
| 2025-03-04 |       2984 |          18 |   12415   |      0.00603217 |                      0 |                      0 |
| 2025-03-05 |       2377 |          20 |   20714   |      0.00841397 |                      0 |                      0 |
| 2025-03-06 |       2702 |          21 |   21754   |      0.00777202 |                      0 |                      0 |
| 2025-03-07 |       2467 |          16 |   18331   |      0.00648561 |                      0 |                      0 |
| 2025-03-08 |       2917 |          20 |   21286   |      0.00685636 |                      0 |                      0 |

## Alerts

- **P1_HIGH** 2025-02-28 — revenue: 1.867e+04 (baseline median 1.4e+04, z=6.89)
  - revenue deviated materially vs prior 7 days (median=14000.00).
- **P0_BLOCKER** 2025-03-01 — revenue: 2.959e+04 (baseline median 1.633e+04, z=3.43)
  - revenue deviated materially vs prior 7 days (median=16334.50).
- **P1_HIGH** 2025-03-02 — revenue: 2.822e+04 (baseline median 1.867e+04, z=1.35)
  - revenue deviated materially vs prior 7 days (median=18669.00).

## Data Quality Gate summary (Part 1)

- Partitions checked: 14
- Total P0 blockers (partition-level): 4

**Blocked / at-risk partitions (P0):**
- 20250226 (events_20250226.csv): score=70
- 20250227 (events_20250227.csv): score=55
- 20250301 (events_20250301.csv): score=70
- 20250308 (events_20250308.csv): score=40

(When a partition is blocked, use quarantine exports to prevent revenue overcounting.)
