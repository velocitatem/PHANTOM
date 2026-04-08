# Configuration reference

This page condenses tables from `[README.md](https://github.com/velocitatem/PHANTOM/blob/main/README.md)` and points to code. Authoritative env templates: `[.env.example](https://github.com/velocitatem/PHANTOM/blob/main/.env.example)`, `[.env.sweep.example](https://github.com/velocitatem/PHANTOM/blob/main/.env.sweep.example)`.

## Core runtime (`.env`)


| Variable                        | Purpose                        | Typical value           |
| ------------------------------- | ------------------------------ | ----------------------- |
| `STORE_MODE`                    | Web mode (`hotel` / `airline`) | `hotel`                 |
| `BACKEND_PORT`                  | Backend API                    | `5000`                  |
| `PROVIDER_PORT`                 | Pricing provider               | `5001`                  |
| `KAFKA_HOST`                    | Kafka broker host              | `localhost`             |
| `KAFKA_PORT`                    | Kafka port                     | `9092`                  |
| `REDIS_PORT`                    | Redis port                     | `6377`                  |
| `REDPANDA_CONSOLE_PORT`         | Kafka UI                       | `8084` (see compose)    |
| `NEXT_PUBLIC_SUPABASE_URL`      | Catalog / data                 | required for full stack |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Catalog / data                 | required                |
| `AIRFLOW_FERNET_KEY`            | Airflow                        | required                |
| `AIRFLOW_SECRET_KEY`            | Airflow web                    | required                |


Web client validation: `[web/src/lib/config.ts](https://github.com/velocitatem/PHANTOM/blob/main/web/src/lib/config.ts)`.

## Training / sweeps (`.env.sweep`)


| Variable        | Purpose                                         |
| --------------- | ----------------------------------------------- |
| `WANDB_API_KEY` | Weights & Biases                                |
| `WANDB_ENTITY`  | Optional override                               |
| `WANDB_PROJECT` | Project name (default `capstone`)               |
| `GITHUB_TOKEN`  | Bootstrap / workers                             |
| `SWEEP_ID`      | Sweep agents (`train.agent`, `benchmark.agent`) |


## Path overrides (`PHANTOM_*`)

Defined in `[lib/config.py](https://github.com/velocitatem/PHANTOM/blob/main/lib/config.py)`:


| Variable                     | Default (conceptual)                |
| ---------------------------- | ----------------------------------- |
| `PHANTOM_DATA_DIR`           | `data/`                             |
| `PHANTOM_EXPERIMENTS_DIR`    | `experiments/`                      |
| `PHANTOM_SIM_RUNS_DIR`       | `sim/rl/runs`                       |
| `PHANTOM_MODEL_REGISTRY_DIR` | `data/models`                       |
| `PHANTOM_COLLECTED_DATA_DIR` | `experiments/agents/collected_data` |


## Makefile entrypoints


| Goal             | Command                                     |
| ---------------- | ------------------------------------------- |
| Platform up/down | `make platform.up` / `make platform.down`   |
| Web dev          | `make web.dev`                              |
| Train            | `make train` (+ `LOCAL_TRAIN_ARGS`)         |
| Benchmark        | `make benchmark` (+ `LOCAL_BENCHMARK_ARGS`) |
| Docs site        | `make docs.platform`                        |


See `make help` for the full list.