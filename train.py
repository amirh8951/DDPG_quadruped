"""
train.py – DDPG training script for UnitreeA1 (position‑control)
================================================================
Run:
    $ python train.py
    $ tensorboard --logdir runs
"""

from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadruped.pybullet_env import PyBulletQuadrupedEnv

# ------------------------------------------------------------
# 1)  Environment
# ------------------------------------------------------------
URDF_PATH = os.path.join(os.path.dirname(__file__), "unitreea1.urdf")

env = PyBulletQuadrupedEnv(
    urdf_path=URDF_PATH,
    render=False,
    episode_max_steps=400,
)

# ------------------------------------------------------------
# 2)  Action‑space exploration noise (radians)
# ------------------------------------------------------------
noise_std = 0.15       # ±0.15 rad ≈ ±8.5°
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(np.zeros(n_actions), noise_std * np.ones(n_actions))

# ------------------------------------------------------------
# 3)  VecEnv with observation/reward normalisation
# ------------------------------------------------------------
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.99)

# ------------------------------------------------------------
# 4)  Custom TensorBoard callback
# ------------------------------------------------------------
class TBStepCallback(BaseCallback):
    """Logs custom scalars each step & stops after `max_episodes`."""

    def __init__(self, *, max_episodes: int = 5_000, log_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.log_freq = log_freq
        self.episode_cnt = 0
        self.ep_step = 0
        self.t0 = None

    def _on_training_start(self) -> None:
        self.t0 = time.time()

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        # ---- custom logs ----
        for key in ["pos_x", "vx", "vy", "vz", "z", "roll", "pitch", "foot_on", "pos_cmd_rms"]:
            tag = (
                f"perf/{key}" if key in {"pos_x", "vx", "vy", "vz"}
                else f"stab/{key}" if key in {"z", "roll", "pitch"}
                else f"energy/{key}"
            )
            self.logger.record(tag, info[key])
        self.logger.record("train/reward", self.locals["rewards"][0])

        # ---- timing ----
        elapsed = time.time() - self.t0 + 1e-8
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.record("time/fps", self.num_timesteps / elapsed)

        # ---- log flush ----
        self.ep_step += 1
        if self.n_calls % self.log_freq == 0:
            self.logger.dump(step=self.num_timesteps)

        if done:
            self.episode_cnt += 1
            self.logger.record("episode", self.episode_cnt)
            self.logger.record("episode/length", self.ep_step)
            self.logger.record("episode/reward", sum(self.locals["rewards"]))
            self.logger.dump(step=self.num_timesteps)
            self.ep_step = 0

        return True  # no early stop

# ------------------------------------------------------------
# 5)  Create the agent
# ------------------------------------------------------------
model = DDPG(
    policy="MlpPolicy",
    env=vec_env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./runs",
    buffer_size=200_000,
    learning_rate=1e-3,
    batch_size=256,
    gamma=0.99,
    tau=1e-3,
)

# ------------------------------------------------------------
# 6)  Train
# ------------------------------------------------------------
model.learn(total_timesteps=1_000_000, callback=TBStepCallback(log_freq=1))

# ------------------------------------------------------------
# 7)  Save model **و** نرمالایزر
# ------------------------------------------------------------
model.save("ddpg_quadruped_posctrl_1M")
vec_env.save("vec_normalize_1M.pkl")

env.close()
