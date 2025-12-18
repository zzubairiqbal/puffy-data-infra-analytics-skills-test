# Data Quality Report

## Summary

- Partitions checked: **14**
- Total failed checks: **18**
- P0 blockers: **4**
- Avg DQ score: **76.4/100**

## Per-partition

- 20250223 (events_20250223.csv): decision=PASS score=100, failed=0 (P0=0, P1=0, P2=0)
- 20250224 (events_20250224.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250225 (events_20250225.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250226 (events_20250226.csv): decision=QUARANTINE score=70, failed=1 (P0=1, P1=0, P2=0)
- 20250227 (events_20250227.csv): decision=QUARANTINE score=55, failed=2 (P0=1, P1=1, P2=0)
- 20250228 (events_20250228.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250301 (events_20250301.csv): decision=QUARANTINE score=70, failed=1 (P0=1, P1=0, P2=0)
- 20250302 (events_20250302.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250303 (events_20250303.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250304 (events_20250304.csv): decision=WARN score=70, failed=2 (P0=0, P1=2, P2=0)
- 20250305 (events_20250305.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250306 (events_20250306.csv): decision=WARN score=70, failed=2 (P0=0, P1=2, P2=0)
- 20250307 (events_20250307.csv): decision=WARN score=85, failed=1 (P0=0, P1=1, P2=0)
- 20250308 (events_20250308.csv): decision=QUARANTINE score=40, failed=3 (P0=1, P1=2, P2=0)

## Cross-partition drift / anomaly signals

- No cross-partition anomalies detected.
