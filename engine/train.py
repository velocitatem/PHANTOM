import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from .wrapper import PHANTOM
from .lib import EconomicMetricsWrapper, MetricsCallback

wandb.init(
    project="phantom-pricing",
    config={"alpha": 0.3, "n_products": 10, "total_timesteps": 50000}
)

env = EconomicMetricsWrapper(PHANTOM(n_products=10, alpha=0.3, render_mode=None))
eval_env = EconomicMetricsWrapper(PHANTOM(n_products=10, alpha=0.3, render_mode=None))

model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=50000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)

metrics_cb = MetricsCallback(log_histograms=True, log_freq=100)
eval_cb = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1)

model.learn(total_timesteps=50000, callback=[metrics_cb, eval_cb])
model.save("phantom_sac")
wandb.finish()

# test trained policy
env = PHANTOM(n_products=10, alpha=0.3, render_mode=None)
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    env.render()
    if term or trunc: break
env.close()
