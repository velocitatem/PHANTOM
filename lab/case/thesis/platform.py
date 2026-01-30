"""Thesis platform with real MDP behavioral models and separability scoring."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ...outlet import (Platform, PlatformConfig, PositionModel, PositionConfig,
                       PostedPriceMechanism, make_instruments, InstrumentType, LogLevel)
from ...outlet.mechanisms.posted_price import PostedPriceConfig
from ...outlet.observation import DefaultObservationBuilder, ObservationConfig
from .arrivals import ContaminatedArrivalModel, ContaminatedArrivalConfig
from .execution import HybridExecutionModel, HybridExecutionConfig
from .objectives import RobustStackelbergObjective, RobustObjectiveConfig


@dataclass
class ThesisConfig:
    # instruments
    n_instruments: int = 10
    cost_range: tuple[float, float] = (5.0, 50.0)
    margin_range: tuple[float, float] = (0.2, 0.5)

    # contamination (Section 3.1)
    alpha_contamination: float = 0.2
    alpha_drift: float = 0.0
    alpha_bounds: tuple[float, float] = (0.0, 0.5)

    # objectives (Eq 23)
    lambda_coi: float = 0.5
    lambda_ux: float = 0.1
    lambda_volatility: float = 0.2
    wasserstein_epsilon: float = 0.1

    # arrivals
    sessions_per_step: int = 30
    human_views_range: tuple[int, int] = (1, 4)
    agent_views_range: tuple[int, int] = (3, 10)

    # inventory
    initial_inventory: float = 100.0
    holding_cost_rate: float = 0.002

    # real behavioral models (from sim.rl)
    use_real_behavior: bool = True
    use_separability: bool = False  # disabled until classifier trained
    human_data_dir: str = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data"
    agent_data_dir: str = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data"

    # simulation
    max_steps: int = 500
    seed: int | None = 24
    log_level: LogLevel = LogLevel.AGG_ONLY


def _resolve_data_dirs(cfg: ThesisConfig) -> tuple[str, str]:
    """Resolve data directories for behavioral models."""
    base = Path(__file__).parent.parent.parent.parent / "experiments"
    human = cfg.human_data_dir or str(base / "collected_data")
    agent = cfg.agent_data_dir or str(base / "agents/collected_data")
    return human, agent


def make_thesis_platform(cfg: ThesisConfig | None = None) -> Platform:
    """Create platform with real MDP behavioral models.

    Implements:
    - Contaminated arrivals using learned MDP kernels from behavior_loader
    - Hybrid execution with real separability scoring from lib.separability
    - Robust Stackelberg objective (Eq 23)
    """
    cfg = cfg or ThesisConfig()
    rng = np.random.default_rng(cfg.seed)
    human_dir, agent_dir = _resolve_data_dirs(cfg)

    instruments = make_instruments(
        n=cfg.n_instruments, cost_range=cfg.cost_range, margin_range=cfg.margin_range,
        inst_type=InstrumentType.SKU, rng=rng)
    instruments.position = np.full(cfg.n_instruments, cfg.initial_inventory)

    arrival = ContaminatedArrivalModel(ContaminatedArrivalConfig(
        base_rate=cfg.sessions_per_step,
        alpha_contamination=cfg.alpha_contamination,
        alpha_drift=cfg.alpha_drift,
        alpha_bounds=cfg.alpha_bounds,
        human_views_range=cfg.human_views_range,
        agent_views_range=cfg.agent_views_range,
        use_real_behavior=cfg.use_real_behavior,
        human_data_dir=human_dir,
        agent_data_dir=agent_dir,
    ))

    execution = HybridExecutionModel(HybridExecutionConfig(
        use_separability=cfg.use_separability,
    ))

    mechanism = PostedPriceMechanism(PostedPriceConfig(max_delta_pct=0.15, min_margin_pct=0.05))
    position = PositionModel(PositionConfig(initial_position=cfg.initial_inventory, holding_cost_rate=cfg.holding_cost_rate))

    market = None
    objective = RobustStackelbergObjective(RobustObjectiveConfig(
        lambda_coi=cfg.lambda_coi, lambda_ux=cfg.lambda_ux,
        lambda_volatility=cfg.lambda_volatility, wasserstein_epsilon=cfg.wasserstein_epsilon))

    obs_builder = DefaultObservationBuilder(ObservationConfig(mask_true_demand=True))
    platform_cfg = PlatformConfig(n_instruments=cfg.n_instruments, max_steps=cfg.max_steps,
                                   seed=cfg.seed, log_level=cfg.log_level, mask_demand=True)

    return Platform(instruments=instruments, mechanism=mechanism, arrival=arrival, execution=execution,
                    position=position, market=market, obs_builder=obs_builder, objective=objective, cfg=platform_cfg)


@dataclass
class AblationConfig(ThesisConfig):
    disable_coi_penalty: bool = False
    disable_ux_penalty: bool = False
    disable_contamination: bool = False
    disable_real_behavior: bool = False


def make_ablation_platform(cfg: AblationConfig) -> Platform:
    if cfg.disable_coi_penalty:
        cfg.lambda_coi = 0.0
    if cfg.disable_ux_penalty:
        cfg.lambda_ux = 0.0
    if cfg.disable_contamination:
        cfg.alpha_contamination = 0.0
    if cfg.disable_real_behavior:
        cfg.use_real_behavior = False
        cfg.use_separability = False
    return make_thesis_platform(cfg)


def sweep_contamination(alpha_values: list[float], base_cfg: ThesisConfig | None = None,
                        n_steps: int = 100, seed: int = 42) -> dict[float, dict]:
    """Test performance across contamination levels (Theorem 1 validation)."""
    from ...experiments.eval import rollout, fixed_price_policy

    results = {}
    base_cfg = base_cfg or ThesisConfig()

    for alpha in alpha_values:
        cfg = ThesisConfig(**{k: v for k, v in base_cfg.__dict__.items() if k != 'alpha_contamination'},
                          alpha_contamination=alpha)
        platform = make_thesis_platform(cfg)
        policy = fixed_price_policy(platform.instruments.refs)
        result = rollout(platform, policy, n_steps, seed=seed)
        results[alpha] = {
            'total_reward': result.total_reward,
            'total_pnl': result.total_pnl,
            'avg_conversion': result.avg_conversion,
            'final_contamination': platform._hidden.contamination,
        }
    return results


def sweep_behavior_modes(base_cfg: ThesisConfig | None = None, n_steps: int = 100, seed: int = 42) -> dict[str, dict]:
    """Compare real vs synthetic behavioral models."""
    from ...experiments.eval import rollout, fixed_price_policy

    base_cfg = base_cfg or ThesisConfig()
    modes = {
        'real_mdp': ThesisConfig(**{**base_cfg.__dict__, 'use_real_behavior': True, 'use_separability': True}),
        'synthetic': ThesisConfig(**{**base_cfg.__dict__, 'use_real_behavior': False, 'use_separability': False}),
        'real_mdp_no_sep': ThesisConfig(**{**base_cfg.__dict__, 'use_real_behavior': True, 'use_separability': False}),
    }

    results = {}
    for name, cfg in modes.items():
        platform = make_thesis_platform(cfg)
        policy = fixed_price_policy(platform.instruments.refs)
        result = rollout(platform, policy, n_steps, seed=seed)
        results[name] = {
            'total_reward': result.total_reward,
            'total_pnl': result.total_pnl,
            'avg_conversion': result.avg_conversion,
        }
    return results
