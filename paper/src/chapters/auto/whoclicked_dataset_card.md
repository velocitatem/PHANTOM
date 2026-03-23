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

<img align="right" width="280" src="https://raw.githubusercontent.com/velocitatem/PHANTOM/main/docs/static/images/banner.svg" alt="PHANTOM research banner" />

# [whoclickedit](https://huggingface.co/datasets/velocitatem/whoclickedit)

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/velocitatem/whoclickedit)
![Rows](https://img.shields.io/badge/Rows-3874-0A9396?style=flat-square)
![Columns](https://img.shields.io/badge/Columns-42-005F73?style=flat-square)
![Sessions](https://img.shields.io/badge/Sessions-36-1D3557?style=flat-square)
![Human rows](https://img.shields.io/badge/Human%20rows-798-2A9D8F?style=flat-square)
![Agent rows](https://img.shields.io/badge/Agent%20rows-3076-E76F51?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-111827?style=flat-square)

> **Event-level behavior data for dynamic pricing research.**
> This dataset captures how humans and automated agents browse, query prices, and move through the PHANTOM storefronts during controlled experiments.

## What this dataset gives you

- A single flat file (`whoclicked.csv`) with both interaction and price-log events.
- Explicit labels for actor origin: `actor_type` and `is_agent`.
- Provenance fields from Kafka envelopes when available.
- Metadata flattened into feature-ready `metadata_*` columns.

## Snapshot

| Metric | Value |
| --- | --- |
| Rows | `3874` |
| Columns | `42` |
| Time range (UTC) | `2025-12-05T09:43:31.301000+00:00` -> `2026-03-23T12:08:30.151000+00:00` |
| Unique sessions | `36` |

## Composition

### Rows by actor
| Actor | Rows | Share |
| --- | --- | --- |
| `human` | 798 | 20.6% |
| `agent` | 3076 | 79.4% |

### Rows by actor and record type
| Actor | Record type | Rows |
| --- | --- | --- |
| `agent` | `interaction` | 197 |
| `agent` | `price_log` | 2879 |
| `human` | `interaction` | 328 |
| `human` | `price_log` | 470 |

### Store mode coverage
| Store mode | Rows |
| --- | --- |
| `hotel` | 3628 |
| `airline` | 196 |
| `shop` | 50 |

### Top interaction events
| Interaction event | Count |
| --- | --- |
| `page_view` | 246 |
| `learn_more_about_item` | 91 |
| `view_item_page` | 88 |
| `add_item_to_cart` | 47 |
| `hover_over_title` | 23 |
| `checkout_start` | 20 |
| `hover_over_paragraph` | 6 |
| `remove_item` | 4 |

## Collection pipeline

Data is sourced from two roots inside PHANTOM:

- `experiments/collected_data` (human sessions)
- `experiments/agents/collected_data` (agent sessions)

Each session directory contains:

- `int.json`: user interaction events
- `price.json`: price quote observations

ETL behavior:

1. Accepts both Kafka-envelope records and flat payload records.
2. Flattens nested JSON to a tabular schema.
3. Preserves row-level provenance (`source_session_dir`, `source_row_index`, topic fields).
4. Adds modeling labels (`actor_type`, `is_agent`, `record_type`).

## Schema highlights

Core modeling fields:

- `actor_type`, `is_agent`, `record_type`
- `sessionId`, `experimentId`, `storeMode`, `ts`
- `eventName`, `page`, `productId`, `price`, `userAgent`

Kafka provenance fields:

- `kafka_partition_id`, `kafka_offset`, `kafka_timestamp_ms`, `kafka_compression`
- `kafka_is_transactional`, `kafka_headers`, `kafka_key_*`, `kafka_value_*`

<details>
<summary>Metadata columns in this release</summary>

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

</details>

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("velocitatem/whoclickedit")
```

Recommended split strategy:

- Prefer session-aware or time-aware splits.
- Do not split rows from the same `sessionId` across train and test.

## Intended use

- Human-vs-agent behavior classification.
- Session-level telemetry modeling for dynamic pricing defenses.
- Robustness experiments under agent-mediated reconnaissance.

## Safety and limitations

- `userAgent` and referrer metadata can be quasi-identifying in very small samples.
- Data comes from a controlled research platform, not a full production marketplace.
- Current release has stronger coverage for `hotel` flows than `airline` flows.

## Citation

If you use this dataset, cite the PHANTOM thesis project and link this page:
`https://huggingface.co/datasets/velocitatem/whoclickedit`
