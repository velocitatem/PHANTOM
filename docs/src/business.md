# Business overview

PHANTOM targets **platform operators and researchers** who need to:

1. **Observe** session-level behavior and price quotes together (trajectories and policies—not just clicks).
2. **Separate** human-driven demand signals from agent-mediated reconnaissance where possible (distinguishability and contamination \alpha in the thesis).
3. **Evaluate** pricing policies that remain useful when **Cost of Information (COI)** is under pressure from automated querying (formal COI framework and theorem in the thesis PDF).

## What this product is not

- A drop-in fraud API that returns “bot score” for every request without your event schema.
- A certified compliance guarantee for regulated pricing: it is a **research stack** with configurable experiments.
- A hosted SaaS: you run the stack (or adapt components) under your infrastructure policy.

## Self-service story (ideal path)

A team connects their **catalog** (today: Supabase-backed flows in this repo), streams **interaction events** through the ingest path, runs **labeled or weak-labeled** human vs agent sessions, estimates **behavioral kernels**, varies **contamination** in simulation, and **trains or benchmarks** robust policies via `engine/`. Steps and caveats are in [Setup](platform-setup.md) (same content as root `SETUP.md`).

## Thesis link

Problem statement, contributions, and research questions: **Introduction** and abstract in the [thesis PDF](https://pub-d5b94a3c29fd40c6b3881946e463fdb7.r2.dev/thesis-latest.pdf).