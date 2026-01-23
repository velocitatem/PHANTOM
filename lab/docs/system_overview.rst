System Overview
===============

The simulator organises dynamic pricing and market-making experiments as a
closed loop with the following stages:

* **Quote** – a policy or agent emits a :class:`lab.outlet.types.Quote`. The
  quote is normalised and validated by a concrete
  :class:`lab.outlet.protocols.Mechanism` implementation
  (posted-price, two-sided, auction).
* **Arrival** – a :class:`lab.outlet.protocols.ArrivalModel` samples a stream of
  :class:`lab.outlet.types.Opportunity` objects given the current time,
  instrument catalogue, and market state.
* **Execution** – the :class:`lab.outlet.protocols.ExecutionModel` converts an
  opportunity into a probabilistic fill using the active quote, optional
  competitor prices, and demand-side context.
* **Position** – a :class:`lab.outlet.protocols.PositionModel` enforces
  inventory or position constraints, censors oversized fills, and accrues
  holding and shortage costs.
* **Observation & Reward** – the
  :class:`lab.outlet.protocols.ObservationBuilder` constructs the censored view
  exposed to the agent, while a :class:`lab.outlet.protocols.Objective`
  transforms :class:`lab.outlet.types.StepMetrics` into a scalar reward with an
  optional breakdown per term.

These components are orchestrated by :class:`lab.outlet.platform.Platform`,
which manages internal hidden state, deterministic seeding, and logging.

Component Matrix
----------------

===============================  ==============================================
Layer                            Responsibilities / Examples
===============================  ==============================================
Mechanisms                       Quote normalisation, execution semantics
                                 (`posted_price`, `two_sided`, `auction`).
Population models                Arrivals (:mod:`lab.population.arrivals`),
                                 execution probability models
                                 (:mod:`lab.population.execution`), and
                                 competitor or market dynamics
                                 (:mod:`lab.population.competitors`).
Position management              Inventory limits, replenishment, holding and
                                 shortage costs (:mod:`lab.outlet.stock`).
Observation & logging            Censored observations and optional event logs
                                 (:mod:`lab.outlet.observation`).
Objectives                       Reward composition utilities
                                 (:mod:`lab.outlet.objectives`).
Experiments                      Rollout helpers, baseline policies, off-policy
                                 evaluation (:mod:`lab.experiments.eval`).
===============================  ==============================================

Preconfigured Platforms
-----------------------

Two high-level factories in :mod:`lab.config` wire common combinations of the
building blocks:

* **Retail dynamic pricing** – posted-price mechanism, session arrivals with
  contamination, elasticity-based executions, reactive competitor model, and a
  composite objective that penalises volatility, holding costs, and lost
  opportunities.
* **Market making** – two-sided quoting, Hawkes order flow, intensity-based
  executions, geometric Brownian motion mid-prices, and an objective combining
  PnL, spread capture, and quadratic inventory risk.

State & Reset Behaviour
-----------------------

When you call :meth:`lab.outlet.platform.Platform.reset`, the platform resets
instrument positions, quotes, and hidden state, but component implementations
may maintain their own internal buffers. For reproducible experiments:

* Reuse freshly instantiated arrival/market models per episode, or add explicit
  ``reset`` methods if the model keeps history (for example,
  :class:`lab.population.arrivals.HawkesArrivalModel` maintains an event
  history, while :class:`lab.population.competitors.ReactiveCompetitorModel`
  tracks prior competitor quotes).
* Seed randomness through the factory configuration (``RetailConfig.seed`` or
  ``MarketMakingConfig.seed``) or pass a seed to ``Platform.reset`` for
  deterministic rollouts.

Extending the Platform
----------------------

To support a new domain:

1. Create custom Mechanism/Arrival/Execution/Market/Observation components by
   implementing the respective protocol in :mod:`lab.outlet.protocols`.
2. Compose a new objective with
   :func:`lab.outlet.objectives.factory.make_composite` or write a bespoke
   :class:`lab.outlet.objectives.base.BaseObjective`.
3. Wire everything together via :class:`lab.outlet.platform.Platform` directly
   or expose a helper factory in :mod:`lab.config`.

Use :func:`lab.experiments.rollout` and
:func:`lab.experiments.compare_policies` to benchmark candidate policies under
multiple random seeds, collecting per-step logs for analysis or OPE.
