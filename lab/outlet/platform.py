"""
Main simulation platform orchestrating the Quote-Control loop.

The Platform class is the central coordinator that:
1. Receives pricing actions (quotes) from the agent
2. Generates arrivals via the ArrivalModel
3. Processes executions via Mechanism and ExecutionModel
4. Applies position censorship via PositionModel
5. Computes metrics and reward via Objective
6. Returns censored observations

Example:
    >>> from lab.config import make_retail_platform
    >>> platform = make_retail_platform()
    >>> result = platform.reset(seed=42)
    >>> result = platform.step(platform.instruments.refs * 1.1)
    >>> print(f"PnL: {result.metrics.pnl:.2f}")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .types import (Quote, Opportunity, Execution, InstrumentSet, StepLogs, StepMetrics,
                    StepEvent, MarketState, HiddenState, Observation, StepResult)
from .constants import LogLevel, EventType, Side
from .protocols import Mechanism, ArrivalModel, ExecutionModel, PositionModel, MarketModel, ObservationBuilder, Objective
from .stock import PositionModel as DefaultPositionModel, PositionConfig
from .observation import DefaultObservationBuilder, ObservationConfig
from .objectives.factory import retail_objective

@dataclass
class PlatformConfig:
    """Configuration for the simulation platform.

    Attributes:
        n_instruments: Number of instruments in the simulation
        max_steps: Maximum steps before episode terminates
        dt: Time duration per step (affects arrival rates)
        log_level: Verbosity of logging (NONE, AGG_ONLY, FULL)
        mask_demand: If True, observations exclude true demand (research mode)
        seed: Random seed for reproducibility
    """
    n_instruments: int = 10
    max_steps: int = 1000
    dt: float = 1.0
    log_level: LogLevel = LogLevel.AGG_ONLY
    mask_demand: bool = True
    seed: int | None = None

class Platform:
    """Main simulation orchestrator implementing Quote -> Arrival -> Execution -> Position.

    The Platform coordinates all components to simulate a pricing environment:
    - Mechanism: validates quotes and determines execution logic
    - ArrivalModel: generates demand opportunities
    - ExecutionModel: computes acceptance probabilities
    - PositionModel: manages inventory/position and censorship
    - MarketModel: updates competitor/market state
    - ObservationBuilder: constructs censored observations
    - Objective: computes reward from metrics

    Attributes:
        instruments: The instrument set being priced
        mechanism: Quote validation and execution mechanism
        arrival: Demand arrival generator
        execution: Acceptance probability model
        position: Inventory/position manager
        market: Competitor/market dynamics (optional)
        obs_builder: Observation constructor
        objective: Reward function
        cfg: Platform configuration
    """

    def __init__(self, instruments: InstrumentSet, mechanism: Mechanism,
                 arrival: ArrivalModel, execution: ExecutionModel,
                 position: PositionModel | None = None,
                 market: MarketModel | None = None,
                 obs_builder: ObservationBuilder | None = None,
                 objective: Objective | None = None,
                 cfg: PlatformConfig | None = None):
        self.instruments = instruments
        self.mechanism = mechanism
        self.arrival = arrival
        self.execution = execution
        self.position = position or DefaultPositionModel(PositionConfig())
        self.market = market
        self.obs_builder = obs_builder or DefaultObservationBuilder()
        self.objective = objective or retail_objective()
        self.cfg = cfg or PlatformConfig(n_instruments=instruments.n)

        self._t: int = 0
        self._rng: np.random.Generator = np.random.default_rng(self.cfg.seed)
        self._quote: Quote | None = None
        self._market_state: MarketState | None = None
        self._hidden: HiddenState = HiddenState()
        self._prev_prices: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> StepResult:
        """Reset the platform to initial state.

        Args:
            seed: Random seed (overrides config seed if provided)

        Returns:
            Initial StepResult with zeroed metrics and initial observation
        """
        self._t = 0
        self._rng = np.random.default_rng(seed or self.cfg.seed)
        self._hidden = HiddenState()
        self._prev_prices = self.instruments.refs.copy()

        # reset position
        self.position.reset(self.instruments, self._rng)
        self.instruments.position = self.position.position

        # initial quote at reference prices
        self._quote = Quote(prices=self.instruments.refs.copy(), propensity=1.0,
                            metadata={'prev_prices': self._prev_prices})
        self._quote = self.mechanism.apply_quote(self._quote, self.instruments, self._rng)

        # initial market state
        if self.market:
            self._market_state = self.market.step(0, self._quote, self._hidden, self._rng)

        # build initial observation
        logs = StepLogs(aggregates={'reset': True},
                        true_demand=np.zeros(self.instruments.n),
                        censored_fills=np.zeros(self.instruments.n))
        metrics = StepMetrics()
        obs = self.obs_builder.build(self._quote, self.instruments, logs, metrics,
                                     self._market_state, self._hidden, self.cfg.mask_demand, 0)

        return StepResult(obs=obs, reward=0.0, terminated=False, truncated=False,
                          info={'true_demand': logs.true_demand}, metrics=metrics,
                          logs=logs, hidden=self._hidden)

    def step(self, action: np.ndarray, propensity: float = 1.0) -> StepResult:
        """Execute one simulation step with the given pricing action.

        The step proceeds as follows:
        1. Apply quote constraints via mechanism
        2. Update market/competitor state
        3. Generate arrivals
        4. Process arrivals -> executions with acceptance check
        5. Apply position censorship to executions
        6. Update position state
        7. Compute metrics (PnL, costs, etc.)
        8. Build logs with propensities
        9. Construct censored observation
        10. Compute reward

        Args:
            action: Price vector for all instruments
            propensity: P(action | behavior policy) for OPE logging

        Returns:
            StepResult containing observation, reward, metrics, logs, and hidden state
        """
        self._t += 1
        cfg = self.cfg

        # 1. apply quote from action
        self._quote = Quote(prices=action, propensity=propensity,
                            metadata={'prev_prices': self._prev_prices})
        self._quote = self.mechanism.apply_quote(self._quote, self.instruments, self._rng)
        self._prev_prices = self._quote.prices.copy()
        self._hidden.quote_history.append(self._quote.prices.copy())

        # 2. update market/competitors
        if self.market:
            self._market_state = self.market.step(self._t, self._quote, self._hidden, self._rng)
            self._hidden.market_history.append(self._market_state)

        # 3. generate arrivals
        opps = self.arrival.sample(self._t, cfg.dt, self.instruments,
                                   self._market_state, self._hidden, self._rng)

        # 4. process opportunities -> executions
        executions: list[Execution] = []
        events: list[StepEvent] = []
        true_demand = np.zeros(self.instruments.n)

        for opp in opps:
            # log exposure
            if cfg.log_level == LogLevel.FULL:
                events.append(StepEvent(t=opp.t, type=EventType.EXPOSURE,
                                        instrument_id=opp.instrument_id,
                                        opportunity_id=opp.id,
                                        price=float(self._quote.prices[opp.instrument_id]),
                                        propensity=self._quote.propensity))

            # check acceptance
            prob = self.execution.prob(opp, self._quote, self.instruments,
                                       self._market_state, self._rng)
            if self._rng.random() < prob:
                # create execution
                exe = self.mechanism.process_opportunity(opp, self._quote, self.instruments,
                                                         self._market_state, self._rng)
                if exe:
                    true_demand[exe.instrument_id] += exe.size_requested
                    # apply position censorship
                    exe = self.position.apply_execution(exe)
                    executions.append(exe)
                    if cfg.log_level == LogLevel.FULL:
                        events.append(StepEvent(t=exe.t, type=EventType.EXECUTION,
                                                instrument_id=exe.instrument_id,
                                                opportunity_id=exe.opportunity_id,
                                                price=exe.price, size=exe.size_filled,
                                                propensity=exe.propensity))

        # 5. update position state
        self.position.step(self._t)
        self.instruments.position = self.position.position

        # 6. compute metrics
        censored_fills = np.zeros(self.instruments.n)
        revenue = 0.0
        cost = 0.0
        spread_capture = 0.0

        for exe in executions:
            censored_fills[exe.instrument_id] += exe.size_filled
            if exe.side == Side.BUY:
                revenue += exe.price * exe.size_filled
                cost += self.instruments.costs[exe.instrument_id] * exe.size_filled
            else:
                revenue -= exe.price * exe.size_filled
                cost -= self.instruments.costs[exe.instrument_id] * exe.size_filled
            # spread capture for market making
            if self._quote.spreads is not None and self._market_state and self._market_state.mid_prices is not None:
                mid = self._market_state.mid_prices[exe.instrument_id]
                if exe.side == Side.BUY:
                    spread_capture += (exe.price - mid) * exe.size_filled
                else:
                    spread_capture += (mid - exe.price) * exe.size_filled

        pnl = revenue - cost
        units = float(np.sum(censored_fills))
        lost = float(np.sum(true_demand - censored_fills))

        # volatility
        volatility = 0.0
        if len(self._hidden.quote_history) > 1:
            prev = self._hidden.quote_history[-2]
            volatility = float(np.mean(np.abs(self._quote.prices - prev) / (prev + 1e-8)))

        metrics = StepMetrics(
            pnl=pnl, revenue=revenue, cost=cost, units_traded=units,
            position_cost=self.position.holding_cost,
            lost_opportunity=self.position.shortage_cost + lost * np.mean(self._quote.prices) * 0.1,
            spread_capture=spread_capture, volatility=volatility,
            conversion=units / (len(opps) + 1e-8),
            per_instrument={'fills': censored_fills, 'demand': true_demand}
        )

        # 7. build logs
        logs = StepLogs(
            events=events if cfg.log_level == LogLevel.FULL else None,
            executions=executions if cfg.log_level == LogLevel.FULL else None,
            aggregates={'n_arrivals': len(opps), 'n_executions': len(executions),
                        'exposures': np.bincount([o.instrument_id for o in opps],
                                                 minlength=self.instruments.n).astype(float)},
            true_demand=true_demand,
            censored_fills=censored_fills
        )

        # 8. build observation
        obs = self.obs_builder.build(self._quote, self.instruments, logs, metrics,
                                     self._market_state, self._hidden, cfg.mask_demand, self._t)

        # 9. compute reward
        reward = self.objective.reward(self._quote, self.instruments, metrics, self._hidden, obs)
        breakdown = self.objective.breakdown(self._quote, self.instruments, metrics, self._hidden, obs)
        # print(f"Step {self._t}: Reward={reward:.2f}, Breakdown={breakdown}")


        # 10. check termination
        terminated = self._t >= cfg.max_steps
        truncated = False

        info = {'true_demand': true_demand, 'breakdown': self.objective.breakdown(
            self._quote, self.instruments, metrics, self._hidden, obs)}

        return StepResult(obs=obs, reward=reward, terminated=terminated, truncated=truncated,
                          info=info, metrics=metrics, logs=logs, hidden=self._hidden)
