# PHANTOM

LLM agents are quietly eroding the pricing power of dynamic pricing systems. They conduct reconnaissance across isolated sessions at machine speed and execute purchases through clean sessions that quote floor prices. The margin that should have accumulated never does.

PHANTOM is a research platform for measuring, simulating, and defending against that erosion. It provides behavioral fingerprinting of human vs agent sessions, a contamination generator for controlled experiments, and a Distributionally Robust RL gym for training pricing policies that hold up under automated demand.

---

## Where to start

| Document | What it covers |
| --- | --- |
| [Business overview](business.md) | The problem, capabilities, and who this is for |
| [Setup](platform-setup.md) | Full bring-up: Docker stack, ingest, behavioral kernels, contamination, RL training |
| [Architecture](architecture.md) | Service map and data flow |
| [Configuration reference](configuration.md) | Env vars, paths, and Makefile targets |
| [Roadmap & notes](roadmap.md) | What is turnkey vs research-grade |

## Key references

- **Thesis PDF:** [thesis-latest.pdf](https://pub-d5b94a3c29fd40c6b3881946e463fdb7.r2.dev/thesis-latest.pdf) — formal model, COI erosion proof, DR-RL formulation
- **Repo root:** [`SETUP.md`](https://github.com/velocitatem/PHANTOM/blob/main/SETUP.md) | [`README.md`](https://github.com/velocitatem/PHANTOM/blob/main/README.md)
- **Academic landing page:** [velocitatem.github.io/PHANTOM/](https://velocitatem.github.io/PHANTOM/)
