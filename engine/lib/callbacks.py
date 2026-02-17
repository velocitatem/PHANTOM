"""Training callbacks for W&B/TensorBoard logging - reads from info dict."""

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

from ..wandb_checkpoint import checkpoint_artifact_name, log_checkpoint_file

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


class CheckpointArtifactCallback(BaseCallback):
    """Periodic SB3 checkpoint uploader backed by W&B artifacts."""

    def __init__(self, cfg: dict, interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = dict(cfg)
        self.interval = max(1, int(interval))
        self.model_dir = Path(str(self.cfg.get("model_dir", "engine/models")))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = self.interval
        self._last_saved_step = -1

    def _artifact_name(self) -> str:
        sweep_id = (
            getattr(wandb.run, "sweep_id", None)
            if HAS_WANDB and wandb.run is not None
            else None
        )
        return checkpoint_artifact_name(self.cfg, backend="sb3", sweep_id=sweep_id)

    def _checkpoint_file(self) -> Path:
        algo = str(self.cfg.get("algo", "model"))
        base = self.model_dir / f"phantom_{algo}_checkpoint"
        self.model.save(str(base))
        return base.with_suffix(".zip")

    def _save_checkpoint(self) -> None:
        if not HAS_WANDB or wandb.run is None:
            return
        step = int(self.num_timesteps)
        if step <= self._last_saved_step:
            return
        checkpoint_path = self._checkpoint_file()
        metadata = {
            "step": step,
            "algo": str(self.cfg.get("algo", "unknown")),
            "sweep_id": getattr(wandb.run, "sweep_id", None),
        }
        saved = log_checkpoint_file(
            self._artifact_name(),
            file_path=checkpoint_path,
            artifact_file_name=checkpoint_path.name,
            metadata=metadata,
        )
        if saved:
            self._last_saved_step = step

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_checkpoint:
            return True
        self._save_checkpoint()
        while self._next_checkpoint <= self.num_timesteps:
            self._next_checkpoint += self.interval
        return True

    def _on_training_end(self) -> None:
        self._save_checkpoint()


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
