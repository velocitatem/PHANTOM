import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

# Add parent directory to path to allow importing engine
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.wrapper import PHANTOM
from engine.lib.wrappers import EconomicMetricsWrapper
from engine.lib.providers import (
    ProviderBenchmark,
    BenchmarkConfig,
    RandomBaseline,
    SurgeBaseline,
)


def env_factory(alpha: float):
    """Creates a wrapped PHANTOM environment for testing at a specific alpha level."""
    # Action levels=9 matches the trained PPO model
    # n_products=8 matches the pretrained model's expectation of Box(16,)
    env = PHANTOM(
        n_products=8,
        alpha=alpha,
        N=100,
        action_levels=9,
        action_scale_low=0.8,
        action_scale_high=1.2,
        max_steps=20,  # Short episodes so simulation goes fast
        robust_points=1,  # disable expensive adversarial lookaheads
        render_mode=None,
    )
    env = EconomicMetricsWrapper(env)
    return FlattenObservation(env)


def main():
    print("Loading pre-trained Robust RL model...")
    model_path = Path(__file__).parent.parent / "models" / "phantom_ppo.zip"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have a trained model before running this script.")
        return

    rl_model = PPO.load(model_path)

    # The action space is Discrete(9). Index 4 is the middle (1.0 scale).
    n_actions = 9
    mid_action = n_actions // 2

    providers = {
        "Static (Base)": lambda obs: mid_action,
        "Random": RandomBaseline(n_actions),
        "Heuristic Surge": SurgeBaseline(
            n_actions, high_threshold=60.0, low_threshold=30.0
        ),
        "Robust RL (PPO)": lambda obs: rl_model.predict(obs, deterministic=True)[0],
    }

    config = BenchmarkConfig(
        n_episodes=10,  # Lower episodes to run faster
        alpha_range=[0.0, 0.5, 1.0],  # Fewer alpha levels
        baseline_name="Static (Base)",
    )

    print(f"\nStarting benchmark across alpha levels: {config.alpha_range}")
    print(
        f"Testing {len(providers)} strategies for {config.n_episodes} episodes each...\n"
    )

    benchmark = ProviderBenchmark(env_factory, providers, config)
    results = benchmark.run()

    # 1. Print tabular results
    df = benchmark.to_dataframe()
    summary = benchmark.summary_table()
    print("\n--- Benchmark Summary Table ---")
    print(summary)

    # 2. Save results to CSV for thesis inclusion
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "provider_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    # 3. Plot the degradation of COI / Revenue as alpha increases
    plt.figure(figsize=(12, 5))

    # Plot 1: Revenue vs Alpha
    plt.subplot(1, 2, 1)
    for name in providers.keys():
        provider_data = df[df["name"] == name]
        plt.plot(
            provider_data["alpha"],
            provider_data["mean_revenue"],
            marker="o",
            label=name,
            linewidth=2,
        )
    plt.title("Revenue under Agent Contamination")
    plt.xlabel("Contamination Level (α)")
    plt.ylabel("Mean Episode Revenue ($)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Plot 2: COI Preservation vs Alpha
    plt.subplot(1, 2, 2)
    for name in providers.keys():
        provider_data = df[df["name"] == name]
        plt.plot(
            provider_data["alpha"],
            provider_data["coi_preserved_pct"],
            marker="s",
            label=name,
            linewidth=2,
        )
    plt.title("Cost of Information (COI) Preservation")
    plt.xlabel("Contamination Level (α)")
    plt.ylabel("COI Preserved (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plot_path = out_dir / "alpha_degradation_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved visualization to {plot_path}")


if __name__ == "__main__":
    main()
