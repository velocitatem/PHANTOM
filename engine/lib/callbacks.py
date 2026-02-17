"""Training callbacks for W&B/TensorBoard logging - reads from info dict."""

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class MetricsCallback(BaseCallback):
    """Training metrics logger - reads info['economics'], logs to W&B."""

    def __init__(
        self, log_histograms: bool = True, log_freq: int = 100, verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_histograms = log_histograms
        self.log_freq = log_freq
        self._episode_revenues: list[float] = []

    def _on_step(self) -> bool:
        if not HAS_WANDB or wandb.run is None:
            return True

        for info in self.locals.get("infos", []):
            if "economics" not in info:
                continue

            econ = info["economics"]
            t = self.num_timesteps

            payload = {
                "economics/revenue": econ["revenue"],
                "economics/margin": econ["margin"],
                "coi/level": econ["coi_level"],
                "economics/regret": econ["regret"],
            }
            if "coi_mix" in econ:
                payload["coi/mix"] = econ["coi_mix"]
            if "coi_base" in econ:
                payload["coi/base"] = econ["coi_base"]
            if "coi_leakage" in econ:
                payload["coi/leakage"] = econ["coi_leakage"]
            if "coi_penalty" in econ:
                payload["coi/penalty"] = econ["coi_penalty"]
            wandb.log(payload, step=t)

            self._episode_revenues.append(econ["revenue"])

        # histograms at log_freq intervals
        if self.log_histograms and self.num_timesteps % self.log_freq == 0:
            for info in self.locals.get("infos", []):
                if "prices" in info:
                    wandb.log(
                        {"distributions/prices": wandb.Histogram(info["prices"])},
                        step=self.num_timesteps,
                    )
                if "demand" in info:
                    wandb.log(
                        {"distributions/demand": wandb.Histogram(info["demand"])},
                        step=self.num_timesteps,
                    )

        return True

    def _on_rollout_end(self) -> None:
        if not HAS_WANDB or wandb.run is None or not self._episode_revenues:
            return
        wandb.log(
            {
                "episode/mean_revenue": np.mean(self._episode_revenues),
                "episode/total_revenue": np.sum(self._episode_revenues),
            },
            step=self.num_timesteps,
        )
        self._episode_revenues = []


class EvalMetricsCallback(EvalCallback):
    """Deterministic evaluation - true performance without exploration noise."""

    def __init__(
        self, eval_env, eval_freq: int = 1000, n_eval_episodes: int = 5, **kwargs
    ):
        super().__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs
        )
        self._eval_revenues: list[float] = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        if not HAS_WANDB or wandb.run is None:
            return result

        # log eval metrics after evaluation runs
        if self.n_calls % self.eval_freq == 0 and hasattr(self, "last_mean_reward"):
            wandb.log(
                {
                    "eval/mean_reward": self.last_mean_reward,
                    "eval/mean_revenue": np.mean(self._eval_revenues)
                    if self._eval_revenues
                    else 0,
                },
                step=self.num_timesteps,
            )
            self._eval_revenues = []

        return result

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        # called after each eval episode
        info = locals_.get("info", {})
        if "economics" in info:
            self._eval_revenues.append(info["economics"]["revenue"])
