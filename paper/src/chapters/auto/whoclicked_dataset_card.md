---
pretty_name: whoclickedit
license: mit
language:
- en
task_categories:
- tabular-classification
task_ids:
- tabular-multi-class-classification
tags:
- e-commerce
- dynamic-pricing
- behavioral-telemetry
- human-vs-agent
- session-data
size_categories:
- 1K<n<10K
---

# Dataset Card for whoclickedit

## Dataset Summary
whoclickedit is an event-level behavioral dataset for human versus agent interaction analysis in dynamic pricing experiments.
It merges interaction logs and price quote logs into one flat CSV (`whoclicked.csv`) with explicit labels for actor type.

## Dataset Snapshot
- Rows: `3838`
- Columns: `42`
- Time range (UTC): `2025-12-05T09:43:31.301000+00:00` to `2026-02-28T19:32:06.444000+00:00`
- Unique sessions by actor:
- `agent`: 7
- `human`: 25
- Rows by actor:
- `agent`: 3076
- `human`: 762
- Rows by record type:
- `price_log`: 3331
- `interaction`: 507
- Rows by actor x record type:
- `agent` / `interaction`: 197
- `agent` / `price_log`: 2879
- `human` / `interaction`: 310
- `human` / `price_log`: 452
- Store modes:
- `hotel`: 3592
- `airline`: 196
- `shop`: 50

## Source and Processing
Data is collected from two local roots in the PHANTOM project:
- `experiments/collected_data` (human sessions)
- `experiments/agents/collected_data` (agent sessions)

Each session folder contains:
- `int.json` (interaction events)
- `price.json` (price quote logs)

The ETL does the following:
- Normalizes both Kafka-envelope and flat payload formats
- Flattens nested metadata fields into `metadata_*` columns
- Preserves all raw rows (no deduplication)
- Adds labels:
  - `actor_type` in `{human, agent}`
  - `is_agent` in `{0, 1}`
  - `record_type` in `{interaction, price_log}`

## Data Fields
Core fields used for modeling:
- `actor_type`, `is_agent`, `record_type`
- `sessionId`, `experimentId`, `storeMode`, `ts`
- `eventName`, `page`, `productId`, `price`, `userAgent`

Kafka provenance fields:
- `kafka_partition_id`, `kafka_offset`, `kafka_timestamp_ms`, `kafka_compression`
- `kafka_is_transactional`, `kafka_headers`, `kafka_key_*`, `kafka_value_*`

Flattened metadata fields currently present:
- `metadata_cabinClass`
- `metadata_dateIndex`
- `metadata_dwellTime`
- `metadata_elementText`
- `metadata_fareRule`
- `metadata_flightType`
- `metadata_itemCount`
- `metadata_nights`
- `metadata_price`
- `metadata_referrer`
- `metadata_roomType`
- `metadata_total`
- `metadata_type`

Top interaction events:
- `page_view`: 236
- `learn_more_about_item`: 88
- `view_item_page`: 85
- `add_item_to_cart`: 46
- `hover_over_title`: 23
- `checkout_start`: 19
- `hover_over_paragraph`: 6
- `remove_item`: 4

## Intended Uses
- Human-vs-agent traffic classification
- Session-level behavioral modeling
- Dynamic pricing robustness analysis under agent-mediated reconnaissance

## Out-of-Scope Uses
- Identity inference or user-level profiling
- Credit, employment, insurance, or legal decision making

## Data Splits
No official train/validation/test split is provided in the current release.
Users should create time-aware or session-aware splits to avoid leakage.

## Privacy and Sensitive Content
- `userAgent` and referrer metadata can be quasi-identifying in small samples.
- Use care before publishing derived artifacts that can re-identify participants.

## Limitations
- Data is generated in a controlled experiment platform, not a full production marketplace.
- Agent traffic currently reflects the configured tasking and browser automation setup.
- Coverage is stronger for `hotel` than `airline` in the current release.

## Citation
If you use this dataset, cite the PHANTOM thesis project and link this dataset page.
