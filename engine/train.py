from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from .wrapper import PHANTOM


class RenderCallback(BaseCallback):
    """Renders environment on every step for live visualization."""
    def __init__(self, env: PHANTOM):
        super().__init__()
        self.env = env

    def _on_step(self) -> bool:
        self.env.render()
        return True


env = PHANTOM(n_products=10, alpha=0.3, render_mode="human")
eval_env = PHANTOM(n_products=10, alpha=0.3, render_mode=None)

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

render_cb = RenderCallback(env)
eval_cb = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1)

model.learn(total_timesteps=50000, callback=[render_cb, eval_cb])
model.save("phantom_sac")

# test trained policy
env = PHANTOM(n_products=10, alpha=0.3, render_mode="human")
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    env.render()
    if term or trunc: break
env.close()
