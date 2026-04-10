# Roadmap & implementation notes

This page is the **honesty pass** from the documentation plan: what clients can expect today versus what remains research-heavy.

## Turnkey in this repository

- **Local stack:** Docker Compose services for backend, Kafka, Redis, Airflow, pricing provider, etc.; Next.js via `make web.dev` (see [Platform setup](platform-setup.md)).
- **Demo verticals:** `hotel` and `airline` storefront modes.
- **Engine:** Benchmarks and training entrypoints (`make train`, `make benchmark`), KL-based agent scoring in `[engine/lib/coi.py](https://github.com/velocitatem/PHANTOM/blob/main/engine/lib/coi.py)`, simulator mixing in `[engine/engine.py](https://github.com/velocitatem/PHANTOM/blob/main/engine/engine.py)`.
- **Orchestration hooks:** Ray/TPU scripts (`submit_ray_job.sh`, `make tpu.ray.`*), W&B sweep agents, Docker trainer publish target.

## Usually requires custom engineering

- **Non-Supabase catalog** or checkout flows without adapting the web + backend contracts.
- **Production SLAs** on Kafka, schema registry, or PII boundaries for your jurisdiction.
- **Tight coupling** to a legacy pricing engine without mapping its API to the provider abstraction.

## Thesis vs code

- The **thesis** states theorems and constructions (COI erosion, kernels, \mathcal{G}(\alpha), DR-RL).  
- The **codebase** implements a **subset** of that story for experiments: verify CLI flags and simulator assumptions before claiming 1:1 equivalence with every equation.
- **Catalog-scale kernel expansion** is discussed in **Chapter 3** with explicit validation caveats—do not assume row-stochasticity and Markov structure are automatically preserved at full product cardinality without review.

## Suggested client messaging

Position PHANTOM as a **reproducible research and evaluation stack** for agent-aware pricing, with a path to custom integration—not as a black-box “turn on anti-agent pricing” product without data and engineering investment.