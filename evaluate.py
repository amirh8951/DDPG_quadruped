"""
evaluate.py – Visualise the trained DDPG policy (position‑control)
================================================================
Run:
    $ python evaluate.py
"""

from __future__ import annotations

import os
import time
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadruped.pybullet_env import PyBulletQuadrupedEnv

# --------------------------------------------------------------------------
# 1. Paths
MODEL_PATH = "ddpg_quadruped_posctrl_1M.zip"   # اسم فایل ذخیره‌شده در train.py
NORM_PATH  = "vec_normalize_1M.pkl"  # فایل نرمالایزر
URDF_PATH  = os.path.join(os.path.dirname(__file__), "unitreea1.urdf")

# --------------------------------------------------------------------------
# 2. Environment (GUI)
base_env = PyBulletQuadrupedEnv(
    urdf_path=URDF_PATH,
    render=True,
    frame_skip=4,
)

# ---------- wrap with same VecNormalize used in training ----------
dummy = DummyVecEnv([lambda: base_env])
vec_env = VecNormalize.load(NORM_PATH, dummy)
vec_env.training = False      # freeze statistics
vec_env.norm_reward = False

# --------------------------------------------------------------------------
# 3. Load trained agent
model = DDPG.load(MODEL_PATH, env=vec_env)

# --------------------------------------------------------------------------
# 4. Rollout episodes
EPISODES = 5
for ep in range(EPISODES):
    obs = vec_env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action = np.zeros(shape=(12, ))
        obs, reward, done, info = vec_env.step(action)
        ep_reward += reward[0]
        time.sleep(1.0 / 60.0)   # ~60Hz real‑time

    print(f"Episode {ep + 1}: total reward = {ep_reward:.3f}")

vec_env.close()
