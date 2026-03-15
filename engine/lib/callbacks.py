"""Training callbacks with algorithm-agnostic metric extraction."""

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

from ..telemetry.wandb import get_wandb_module


class MetricsCallback(BaseCallback):
    """Collects interval train metrics from env info dictionaries."""

    def __init__(
        self,
        log_histograms: bool = False,
        log_freq: int = 100,
        hist_freq: int = 500,
        step_offset: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_histograms = log_histograms
        self.log_freq = max(1, int(log_freq))
        self.hist_freq = max(1, int(hist_freq))
        self.step_offset = max(0, int(step_offset))
        self._wandb = get_wandb_module()
        self._wandb_live = bool(self._wandb is not None and self._wandb.run is not None)
        self._price_samples: list[float] = []
        self._demand_samples: list[float] = []
        self._window_sums = {
            "train/revenue_mean": 0.0,
            "train/margin_mean": 0.0,
            "train/coi_level_mean": 0.0,
            "train/regret_mean": 0.0,
            "train/profit_mean": 0.0,
            "train/agent_prob": 0.0,
            "train/alpha_adv": 0.0,
            "train/ux_penalty": 0.0,
            "train/volatility": 0.0,
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
        if "profit" in econ:
            self._window_sums["train/profit_mean"] += float(econ.get("profit", 0.0))
        if "agent_prob" in econ:
            self._window_sums["train/agent_prob"] += float(econ.get("agent_prob", 0.0))
        if "alpha_adv" in econ:
            self._window_sums["train/alpha_adv"] += float(econ.get("alpha_adv", 0.0))
        if "ux_penalty" in econ:
            self._window_sums["train/ux_penalty"] += float(econ.get("ux_penalty", 0.0))
        if "volatility" in econ:
            self._window_sums["train/volatility"] += float(econ.get("volatility", 0.0))
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

    def _accumulate_histograms(self, info: dict[str, Any]) -> None:
        if not self.log_histograms:
            return

        for key in ("effective_prices", "prices"):
            if key not in info:
                continue
            try:
                values = np.asarray(info.get(key), dtype=float).reshape(-1)
            except Exception:
                continue
            if values.size <= 0:
                continue
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                self._price_samples.extend(finite_values.tolist())
            break

        if "demand" in info:
            try:
                demand_values = np.asarray(info.get("demand"), dtype=float).reshape(-1)
            except Exception:
                demand_values = np.array([], dtype=float)
            if demand_values.size > 0:
                finite_demand = demand_values[np.isfinite(demand_values)]
                if finite_demand.size > 0:
                    self._demand_samples.extend(finite_demand.tolist())

    def _flush_histograms(self, step: int, force: bool = False) -> None:
        if not self.log_histograms:
            return
        if not force and step % self.hist_freq != 0:
            return
        if not self._price_samples and not self._demand_samples:
            return
        if self._wandb is None:
            self._price_samples.clear()
            self._demand_samples.clear()
            return

        payload: dict[str, Any] = {}
        if self._price_samples:
            payload["train/price_dist"] = self._wandb.Histogram(
                np.asarray(self._price_samples, dtype=np.float32)
            )
        if self._demand_samples:
            payload["train/demand_dist"] = self._wandb.Histogram(
                np.asarray(self._demand_samples, dtype=np.float32)
            )

        if payload and self._wandb_live:
            try:
                self._wandb.log(payload, step=self.step_offset + int(step))
            except Exception:
                self._wandb_live = False

        self._price_samples.clear()
        self._demand_samples.clear()

    def _flush(self, step: int, *, force_hist: bool = False) -> None:
        if self._window_count > 0:
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
            if self._wandb_live:
                try:
                    self._wandb.log(dict(payload), step=self.step_offset + int(step))
                except Exception:
                    self._wandb_live = False
                    self.events.append(payload)
            else:
                self.events.append(payload)
            for key in self._window_sums:
                self._window_sums[key] = 0.0
            self._window_count = 0

        self._flush_histograms(step=step, force=force_hist)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict):
                self._accumulate(info)
                self._accumulate_histograms(info)

        if self.num_timesteps % self.log_freq == 0:
            self._flush(step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self._flush(step=self.num_timesteps, force_hist=True)


class EvalMetricsCallback(EvalCallback):
    """Deterministic evaluation collector detached from logging backends."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        step_offset: int = 0,
        **kwargs,
    ):
        super().__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs
        )
        self.step_offset = max(0, int(step_offset))
        self._wandb = get_wandb_module()
        self._wandb_live = bool(self._wandb is not None and self._wandb.run is not None)
        self._eval_stats: dict[str, list[float]] = {
            "eval/revenue_mean": [],
            "eval/margin_mean": [],
            "eval/coi_level_mean": [],
            "eval/coi_leakage_mean": [],
            "eval/volatility_mean": [],
            "eval/agent_prob_mean": [],
        }
        self.events: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0 and hasattr(self, "last_mean_reward"):
            payload: dict[str, float | int] = {
                "eval/reward_mean": float(self.last_mean_reward),
                "train/global_step": int(self.num_timesteps),
            }
            for key, values in self._eval_stats.items():
                payload[key] = float(np.mean(values)) if values else 0.0

            if self._wandb_live:
                try:
                    self._wandb.log(
                        dict(payload),
                        step=self.step_offset + int(self.num_timesteps),
                    )
                except Exception:
                    self._wandb_live = False
                    self.events.append(payload)
            else:
                self.events.append(payload)

            for values in self._eval_stats.values():
                values.clear()

        return result

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        # called after each eval episode
        info = locals_.get("info", {})
        econ = info.get("economics") if isinstance(info, dict) else None
        if not isinstance(econ, dict):
            return

        self._eval_stats["eval/revenue_mean"].append(float(econ.get("revenue", 0.0)))
        self._eval_stats["eval/margin_mean"].append(float(econ.get("margin", 0.0)))
        self._eval_stats["eval/coi_level_mean"].append(
            float(econ.get("coi_level", 0.0))
        )
        self._eval_stats["eval/coi_leakage_mean"].append(
            float(econ.get("coi_leakage", 0.0))
        )
        self._eval_stats["eval/volatility_mean"].append(
            float(econ.get("volatility", 0.0))
        )
        self._eval_stats["eval/agent_prob_mean"].append(
            float(econ.get("agent_prob", 0.0))
        )
