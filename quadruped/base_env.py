"""
Base environment for a 12-DOF quadruped (Unitree A1).

Changes for **position-control**:
  • Action space = desired joint angles (rad), bounded exactly by the
    URDF limits of each revolute joint.
  • `_standardize_action()` now simply clips to those bounds – no torque
    scaling.

Child classes must implement reset(), step() and close().
"""

from __future__ import annotations

import gym
from gym import spaces
import numpy as np


class BaseQuadrupedEnv(gym.Env):
    """Abstract parent environment for a 12-DOF quadruped."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Joint-angle limits (radians) copied from the Unitree A1 URDF
    _HIP_LIMITS = (-0.802851455917, 0.802851455917)
    _THIGH_LIMITS = (-1.0471975512, 4.18879020479)
    _CALF_LIMITS = (-2.69653369433, -0.916297857297)

    def __init__(self, n_joints: int = 12):
        super().__init__()

        # ----- configuration ------------------------------------------------
        self.n_joints = n_joints                 # 4 legs × 3 joints each
        self.obs_dim = 49                        # 49-D observation
        # --------------------------------------------------------------------

        # Assemble per-joint limits in the same order as self.joint_ids
        lows = np.array([
            self._HIP_LIMITS[0], self._THIGH_LIMITS[0], self._CALF_LIMITS[0],  # FR
            self._HIP_LIMITS[0], self._THIGH_LIMITS[0], self._CALF_LIMITS[0],  # FL
            self._HIP_LIMITS[0], self._THIGH_LIMITS[0], self._CALF_LIMITS[0],  # RR
            self._HIP_LIMITS[0], self._THIGH_LIMITS[0], self._CALF_LIMITS[0],  # RL
        ], dtype=np.float32)

        highs = np.array([
            self._HIP_LIMITS[1], self._THIGH_LIMITS[1], self._CALF_LIMITS[1],  # FR
            self._HIP_LIMITS[1], self._THIGH_LIMITS[1], self._CALF_LIMITS[1],  # FL
            self._HIP_LIMITS[1], self._THIGH_LIMITS[1], self._CALF_LIMITS[1],  # RR
            self._HIP_LIMITS[1], self._THIGH_LIMITS[1], self._CALF_LIMITS[1],  # RL
        ], dtype=np.float32)

        # Action space: desired joint angles (rad)
        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

        # Observation space placeholder
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Stores last commanded position (needed for obs & reward)
        self._last_action = np.zeros(self.n_joints, dtype=np.float32)

    # ---------- helpers that child classes can reuse -----------------------
    def _standardize_action(self, action: np.ndarray) -> np.ndarray:
        """Clip raw action to the physical joint limits (safety layer)."""
        return np.clip(action, self.action_space.low, self.action_space.high)

    # ---------- abstract interface -----------------------------------------
    def reset(self):
        """Return the first observation (np.float32, shape=obs_dim)."""
        raise NotImplementedError

    def step(self, action):
        """Gym step – must be implemented by subclass."""
        raise NotImplementedError

    def close(self):
        """Release simulation resources (if any)."""
        raise NotImplementedError
