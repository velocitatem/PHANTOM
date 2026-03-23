"""Provider benchmarking - compare pricing strategies across contamination levels."""

from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np
import pandas as pd

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class RandomBaseline:
    """uniform random action selection as a lower-bound baseline"""

    def __init__(self, n_actions: int):
        self.n = n_actions

    def __call__(self, obs):
        return int(np.random.randint(self.n))

    def predict(self, obs, **kw):
        return self(obs), None


class SurgeBaseline:
    """heuristic surge pricing: boost price when demand is above threshold, discount when below.
    matches the naive pricing rule from thesis Section 3.3.2"""

    def __init__(
        self, n_actions: int, high_threshold: float = 60.0, low_threshold: float = 30.0
    ):
        self.n = n_actions
        self.mid = n_actions // 2  # identity action (scale ~1.0)
        self.high_t = high_threshold
        self.low_t = low_threshold

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        n_prod = len(obs) // 2
        demand_mean = float(np.mean(obs[:n_prod])) if n_prod > 0 else 0.0
        if demand_mean >= self.high_t:
            return min(self.mid + 2, self.n - 1)  # surge: two levels above identity
        if demand_mean <= self.low_t:
            return max(self.mid - 2, 0)  # discount: two levels below identity
        return self.mid  # hold

    def predict(self, obs, **kw):
        return self(obs), None


@dataclass
class ProviderResult:
    """Single benchmark result for one provider at one alpha level."""

    name: str
    alpha: float
    total_revenue: float
    mean_revenue: float
    coi_level: float
    coi_preserved_pct: float  # vs alpha=0 baseline
    margin_integrity: float
    regret: float
    episodes: int


@dataclass
class BenchmarkConfig:
    """Configuration for provider benchmark runs."""

    n_episodes: int = 100
    alpha_range: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5])
    baseline_name: str = "fixed"


class ProviderBenchmark:
    """Compare pricing providers to prove margin preservation across contamination levels.

    Usage:
        def env_factory(alpha):
            return EconomicMetricsWrapper(PHANTOM(alpha=alpha))

        providers = {
            "fixed": lambda obs: np.ones(10) * 50,
            "learned": model.predict,
        }

        benchmark = ProviderBenchmark(env_factory, providers)
        results = benchmark.run()
        print(benchmark.summary_table())
    """

    def __init__(
        self,
        env_factory: Callable[[float], Any],
        providers: dict[str, Callable],
        config: BenchmarkConfig | None = None,
    ):
        self.env_factory = env_factory  # fn(alpha) -> wrapped env
        self.providers = providers  # {name: fn(obs) -> action}
        self.config = config or BenchmarkConfig()
        self.results: list[ProviderResult] = []

    def run(self) -> list[ProviderResult]:
        """Run benchmark across all providers and alpha levels."""
        baseline_coi: dict[str, float] = {}  # {provider: coi at alpha=0}

        for alpha in self.config.alpha_range:
            env = self.env_factory(alpha)

            for name, policy_fn in self.providers.items():
                revenues, coi_levels, margins = [], [], []

                for _ in range(self.config.n_episodes):
                    obs, _ = env.reset()
                    episode_revenue = 0.0
                    done = False

                    while not done:
                        action = policy_fn(obs)
                        # handle sb3 model.predict returning tuple
                        if isinstance(action, tuple):
                            action = action[0]
                        obs, reward, term, trunc, info = env.step(action)
                        done = term or trunc

                        econ = info.get("economics", {})
                        episode_revenue += econ.get("revenue", 0)
                        coi_levels.append(econ.get("coi_level", 0))
                        margins.append(econ.get("margin", 0))

                    revenues.append(episode_revenue)

                mean_coi = np.mean(coi_levels) if coi_levels else 0.0
                if alpha == 0.0:
                    baseline_coi[name] = mean_coi

                base = baseline_coi.get(name, mean_coi)
                coi_preserved = mean_coi / base if base > 0 else 1.0

                result = ProviderResult(
                    name=name,
                    alpha=alpha,
                    total_revenue=float(np.sum(revenues)),
                    mean_revenue=float(np.mean(revenues)),
                    coi_level=mean_coi,
                    coi_preserved_pct=coi_preserved * 100,
                    margin_integrity=float(np.mean(margins)) if margins else 0.0,
                    regret=0.0,  # compute vs optimal if known
                    episodes=self.config.n_episodes,
                )
                self.results.append(result)

                # log to wandb if available
                if HAS_WANDB and wandb.run is not None:
                    try:
                        wandb.log(
                            {
                                f"benchmark/{name}/revenue": result.mean_revenue,
                                f"benchmark/{name}/coi_preserved": result.coi_preserved_pct,
                                f"benchmark/{name}/margin": result.margin_integrity,
                                "benchmark/alpha": alpha,
                            }
                        )
                    except Exception:
                        pass

        return self.results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.__dict__ for r in self.results])

    def summary_table(self) -> pd.DataFrame:
        """Pivot table: providers x alpha with revenue/COI metrics."""
        df = self.to_dataframe()
        return df.pivot_table(
            index="name",
            columns="alpha",
            values=["mean_revenue", "coi_preserved_pct", "margin_integrity"],
            aggfunc="mean",
        )
