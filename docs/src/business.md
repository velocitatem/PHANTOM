# Business overview

Dynamic pricing extracts margin by exploiting the information asymmetry between buyer and seller. When a user browses a flight or hotel across multiple sessions, each interaction accumulates demand signals that push the quoted price upward. That is the mechanism working as intended.

LLM agents break it. An agent can conduct reconnaissance—across dozens of isolated sessions, at machine speed—and then execute a purchase through a clean session that looks like a first-time visitor. The platform sees a low-engagement session and quotes a floor price. The margin that should have been captured, the **Cost of Information (COI)**, vanishes. At scale this is not a theoretical concern; it is a structural leak in any session-based pricing system.

**PHANTOM is a research platform for studying and defending against that leak.**

## Who it is for

| Role | What they get |
|---|---|
| Pricing and revenue researchers | A controlled lab with instrumented human and agent sessions, behavioral kernel estimation, and contamination simulation at configurable levels |
| Platform engineers evaluating agent risk | A concrete pipeline from behavioral event logs to a per-session agent-probability signal, ready to feed into an existing pricing provider |
| RL practitioners | A Distributionally Robust RL gym built on a Wasserstein ambiguity set, with benchmark tiers and sweep tooling out of the box |

## Core capabilities

**Behavioral fingerprinting.** PHANTOM logs interaction trajectories at the event level (action, item, timestamp) and fits separate Markov transition kernels for human and agent sessions via MLE. Per-session divergence scores (Δ_H, Δ_A) and a learned agent-probability signal f(τ) are computed on partial trajectories in real time, giving the pricing layer a continuous signal rather than a binary bot flag.

**Contamination simulation.** The contamination generator G(α) mixes real human trajectories with synthetic agent trajectories at a configurable ratio α. This lets you evaluate pricing robustness across the full spectrum from purely human traffic to fully automated demand, without needing live agent traffic in production.

**Robust policy training.** The defense gym trains pricing policies against the worst-case demand distribution within a Wasserstein ball around the generator's empirical distribution. The reward function penalizes COI leakage (weighted by agent probability) while bounding UX degradation for legitimate users.

## The path from logs to defense

A team: connects their catalog and ingest path → streams interaction events through Kafka → labels or weak-labels sessions → estimates behavioral kernels → varies α in simulation → trains and benchmarks robust policies. The full walkthrough is in [Setup](platform-setup.md).

## Scope and honest caveats

This is a **research stack**, not a hosted service:

- It ships two demo verticals (`hotel`, `airline`); a new catalog requires engineering work on events and reward features.
- Kernel estimates are research-grade until validated on your traffic distribution.
- There is no built-in compliance layer for regulated pricing markets.

The thesis PDF contains the formal proofs, the COI erosion theorem, and the full DR-RL formulation. The code operationalizes those constructs—every term in the reward function maps to something computed from your logs.

**Thesis PDF:** [thesis-latest.pdf](https://pub-d5b94a3c29fd40c6b3881946e463fdb7.r2.dev/thesis-latest.pdf) — Introduction and Chapter 3 cover the problem statement, contributions, and formal model.