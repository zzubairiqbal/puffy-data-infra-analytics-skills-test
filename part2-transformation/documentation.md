## Part 2 - Transformation Pipeline (Sessions + Attribution)

### Objective
Transform cleaned raw events into analytics-ready tables that support:
- user engagement measurement (sessions + funnel)
- channel-level performance analysis
- purchase attribution with a **7-day lookback**, supporting **first-click** and **last-click** models

This pipeline is designed to be reproducible and easy to extend.

---

### Inputs
The pipeline is intended to run on **cleaned exports** produced by Part 1. These inputs already:
- coalesce identity into `client_id_canonical`
- quarantine clearly invalid purchases (e.g., revenue <= 0, empty items)
- quarantine transaction_id collisions

The Part 2 runner can also start from raw exports and trigger the Part 1 cleaning step automatically.

---

### Key design decisions

#### Identity
Primary user identifier: `client_id_canonical` = coalesce(`client_id`, `clientId`)

Rationale: the dataset includes a mid-period column rename; coalescing prevents session breaks and attribution drift.

#### Sessionization
Session definition:
- group events by `client_id_canonical`
- sort by timestamp
- a new session starts when inactivity gap > **30 minutes**

Outputs:
- `sessions.csv` contains session-level attributes and derived fields
- `sessionized_events.csv` contains every event with an assigned `session_id`

#### Channel classification
Channel is derived from (in priority order):
1) click IDs / query params on landing page (e.g., `gclid`, `fbclid`)
2) UTM params (`utm_source`, `utm_medium`, `utm_campaign`)
3) referrer domain (organic search / social / referral)
4) direct (no signal)

We store both a normalized `landing_channel` and a `landing_channel_detail` for debugging.

#### Purchases table
A purchase is a `checkout_completed` event. We extract:
- `transaction_id`
- `revenue`
- `user_email` (hashed)
- `items_json`

Purchases are linked to sessions using the sessionized events (purchase event belongs to the session it occurred in).

#### Attribution (7-day lookback)
For each purchase, we look back **7 days** within the same user and select sessions:

- **First-click**: earliest session in the lookback window
- **Last-click**: latest session in the lookback window

Additionally, because many ecommerce teams prefer it, we also compute:
- **Last non-direct**: most recent session where channel != direct (when available)

All three are included in `purchase_attribution.csv`. The required models (first/last) are always present.

---

### Validation / reconciliation
We include a reconciliation file:
- `validation/reconciliation.json`

It verifies:
- purchase counts reconcile to input purchase events
- revenue sums are consistent
- how many purchases lack an assigned session (should be ~0 with clean inputs)

---

### Outputs (for Part 3)
- `sessions.csv`
- `purchases.csv`
- `purchase_attribution.csv`
- `funnel_daily_by_channel.csv`
- `attribution_first_click_daily.csv`
- `attribution_last_click_daily.csv`
- `attribution_last_non_direct_daily.csv`
