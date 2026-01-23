# MOS (Money Operating System)

Research-grade quote-control simulator for studying dynamic pricing and market making policies.
The system models pricing as a closed loop of **Quote → Arrival → Execution → Position**, enabling
controlled experimentation with demand models, inventory constraints, and reward shaping.

## Core Loop

1. **Quote** – the policy posts prices (one-sided or two-sided depending on the mechanism).
2. **Arrival** – a population model generates purchase opportunities or market orders.
3. **Execution** – an execution model decides whether an arrival converts at the quoted price.
4. **Position** – inventory/position limits censor fills and generate holding/shortage costs.
5. **Observation & Reward** – censored fills and aggregate metrics are exposed to the agent, while
   objectives turn metrics into a scalar reward.

Each stage is pluggable via light-weight protocols so you can swap in alternative mechanisms,
demand models, or objectives without rewriting the rest of the simulator.

## Package Layout

| Module            | Purpose |
|-------------------|---------|
| `lab.outlet`      | Core simulation engine, domain types, pricing mechanisms, objectives. |
| `lab.population`  | Demand arrival models, execution probability models, competitor/market dynamics. |
| `lab.experiments` | Rollout utilities, baseline policies, and off-policy evaluation helpers. |
| `lab.config`      | Convenience factories for preconfigured retail and market-making environments. |

## Preconfigured Scenarios

### Retail Dynamic Pricing
- Mechanism: posted prices with margin and delta constraints.
- Arrivals: browsing sessions with contamination support (scrapers).
- Execution: elasticity model with competitor cross-effects.
- Position: inventory tracking with holding and shortage costs.
- Market: reactive competitor that can trigger price wars.
- Objective: PnL minus volatility, holding cost, and lost opportunity penalties.

```python
from lab.config import make_retail_platform
from lab.experiments import rollout, fixed_price_policy

platform = make_retail_platform()
policy = fixed_price_policy(platform.instruments.refs)
result = rollout(platform, policy, n_steps=100)
print(result.total_pnl)
```

### Market Making
- Mechanism: two-sided quoting with bid/ask spreads.
- Arrivals: Hawkes order flow for clustered demand.
- Execution: Avellaneda–Stoikov style intensity model.
- Position: inventory risk limits and quadratic penalty objective.
- Market: geometric Brownian motion mid-price process.
- Objective: PnL plus spread capture minus inventory risk.

```python
from lab.config import make_market_making_platform
from lab.experiments import rollout

platform = make_market_making_platform()
mm_policy = lambda obs, t: (platform.instruments.refs, 1.0)
result = rollout(platform, mm_policy, n_steps=200, seed=42)
print(result.total_pnl)
```

## Extending the Simulator

- Implement `lab.outlet.protocols.Mechanism` or `ArrivalModel` to introduce new pricing
domains or demand processes.
- Compose objectives with `lab.outlet.objectives.factory.make_composite` to study alternate
reward formulations.
- Use `lab.experiments.compare_policies` to benchmark candidate policies across multiple
random seeds.

Comprehensive API documentation lives in `lab/docs` (build with `make html`).
