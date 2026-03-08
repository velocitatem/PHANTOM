"""Training callbacks with algorithm-agnostic metric extraction."""

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np


class MetricsCallback(BaseCallback):
    """Collects interval train metrics from env info dictionaries."""

    def __init__(
        self,
        log_histograms: bool = False,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_histograms = log_histograms
        self.log_freq = max(1, int(log_freq))
        self._window_sums = {
            "train/revenue_mean": 0.0,
            "train/margin_mean": 0.0,
            "train/coi_level_mean": 0.0,
            "train/regret_mean": 0.0,
            "train/coi_mix": 0.0,
            "train/coi_base": 0.0,
            "train/coi_leakage": 0.0,
            "train/coi_penalty": 0.0,
        }
        self._window_count = 0
        self.events: list[dict[str, Any]] = []

    def _accumulate(self, info: dict[str, Any]) -> None:
        econ = info.get("economics")
        if not isinstance(econ, dict):
            return
        self._window_sums["train/revenue_mean"] += float(econ.get("revenue", 0.0))
        self._window_sums["train/margin_mean"] += float(econ.get("margin", 0.0))
        self._window_sums["train/coi_level_mean"] += float(econ.get("coi_level", 0.0))
        self._window_sums["train/regret_mean"] += float(econ.get("regret", 0.0))
        if "coi_mix" in econ:
            self._window_sums["train/coi_mix"] += float(econ.get("coi_mix", 0.0))
        if "coi_base" in econ:
            self._window_sums["train/coi_base"] += float(econ.get("coi_base", 0.0))
        if "coi_leakage" in econ:
            self._window_sums["train/coi_leakage"] += float(
                econ.get("coi_leakage", 0.0)
            )
        if "coi_penalty" in econ:
            self._window_sums["train/coi_penalty"] += float(
                econ.get("coi_penalty", 0.0)
            )
        self._window_count += 1

    def _flush(self, step: int) -> None:
        if self._window_count <= 0:
            return
        denom = float(self._window_count)
        payload = {
            key: (value / denom)
            for key, value in self._window_sums.items()
            if value != 0.0
            or key
            in {
                "train/revenue_mean",
                "train/margin_mean",
                "train/coi_level_mean",
                "train/regret_mean",
            }
        }
        payload["train/global_step"] = int(step)
        self.events.append(payload)
        for key in self._window_sums:
            self._window_sums[key] = 0.0
        self._window_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict):
                self._accumulate(info)

        if self.num_timesteps % self.log_freq == 0:
            self._flush(step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self._flush(step=self.num_timesteps)


class EvalMetricsCallback(EvalCallback):
    """Deterministic evaluation collector detached from logging backends."""

    def __init__(
        self, eval_env, eval_freq: int = 1000, n_eval_episodes: int = 5, **kwargs
    ):
        super().__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs
        )
        self._eval_revenues: list[float] = []
        self.events: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0 and hasattr(self, "last_mean_reward"):
            self.events.append(
                {
                    "eval/reward_mean": float(self.last_mean_reward),
                    "eval/revenue_mean": float(np.mean(self._eval_revenues))
                    if self._eval_revenues
                    else 0.0,
                    "train/global_step": int(self.num_timesteps),
                }
            )
            self._eval_revenues = []

        return result

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        # called after each eval episode
        info = locals_.get("info", {})
        if "economics" in info:
            self._eval_revenues.append(info["economics"]["revenue"])
