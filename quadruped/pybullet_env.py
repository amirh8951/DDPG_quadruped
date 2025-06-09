"""
PyBullet implementation of the UnitreeA1 environment (position control).

Observation layout (49):
    0–11   joint positions
    12–23  joint velocities
    24–26  base position (x, y, z)
    27–29  base linear velocity (vx, vy, vz)
    30–32  base Euler angles   (roll, pitch, yaw)
    33–36  foot‑contact flags  (FR, FL, RR, RL)  ∈ {0,1}
    37–48  last commanded joint positions
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pybullet as p
import pybullet_data

from .base_env import BaseQuadrupedEnv


class PyBulletQuadrupedEnv(BaseQuadrupedEnv):

    FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

    # ------------------------------------------------------------------ #
    #  Constructor                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        urdf_path: str = "unitreea1.urdf",
        render: bool = False,
        frame_skip: int = 4,
        episode_max_steps: int = 400,
    ):
        super().__init__(n_joints=12)

        self.urdf_path = urdf_path
        self.render_mode = render
        self.frame_skip = frame_skip

        self.step_count = 0
        self.max_episode_steps = episode_max_steps

        # Connect to PyBullet
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_id: int | None = None          # set in reset()
        self.foot_link_ids: list[int] = []        # filled after loading URDF
        self.joint_ids: list[int] = []            # ditto
        self._prev_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Gym interface                                                     #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Reset simulation and return first observation."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.3]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_ori)

        # Generic friction/damping tweaks
        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, linearDamping=0.04, angularDamping=0.04)
            p.changeDynamics(self.robot_id, j, lateralFriction=1.0, restitution=0.0)

        # Disable default motors so we have full control
        for j in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(
                self.robot_id, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
            )

        # Build joint/foot ID lists
        self.joint_ids.clear()
        self.foot_link_ids.clear()
        for j in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, j)[12].decode()
            if link_name in self.FOOT_NAMES:
                self.foot_link_ids.append(j)

            joint_name = p.getJointInfo(self.robot_id, j)[1].decode()
            if joint_name.endswith("_joint") and not joint_name.startswith("imu"):
                self.joint_ids.append(j)

        # Preserve the order FR, FL, RR, RL for feet
        self.foot_link_ids = [
            next(j for j in self.foot_link_ids if p.getJointInfo(self.robot_id, j)[12].decode() == n)
            for n in self.FOOT_NAMES
        ]

        self._last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Two warm‑up steps for numerical stability
        for _ in range(2):
            p.stepSimulation()

        self.step_count = 0
        self._prev_obs = None
        return self._get_obs()

    # ------------------------------------------------------------------ #
    def step(self, action):
        """Apply desired joint angles, step physics, compute reward."""
        action = self._standardize_action(action)      # safety‑clip

        # Send position commands (PD‑style)
        for i, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(action[i]),
                positionGain=0.4,
                velocityGain=0.15,
                force=33.5,          # effort limit from URDF
            )

        for _ in range(self.frame_skip):
            # Advance physics (frame_skip internal steps)
            p.stepSimulation()

        obs = self._get_obs()
        rew = self._compute_reward(obs, action)
        done = self._check_termination(obs)

        contacts = obs[33:37]
        info = {
            "pos_x":      obs[24],
            "vx":         obs[27],
            "vy":         obs[28],
            "vz":         obs[29],
            "z":          obs[26],
            "roll":       obs[30],
            "pitch":      obs[31],
            "foot_on":    float((contacts > 0.1).sum()),
            "pos_cmd_rms": float(np.sqrt(np.square(action).mean())),
        }

        self._last_action = action.copy()
        self.step_count += 1
        self._prev_obs = obs.copy()  # keep for Δvx next step
        return obs, rew, done, info

    # ------------------------------------------------------------------ #
    def close(self):
        p.disconnect()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        """Build the 49‑D observation vector."""
        joint_pos, joint_vel = [], []
        for j in self.joint_ids:
            js = p.getJointState(self.robot_id, j)
            joint_pos.append(js[0])
            joint_vel.append(js[1])

        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_ori)
        lin_vel, _ = p.getBaseVelocity(self.robot_id)

        # Foot‑contact flags
        contacts = [
            1.0 if p.getContactPoints(self.robot_id, -1, linkIndexA=lid) else 0.0
            for lid in self.foot_link_ids
        ]

        obs = np.array(
            joint_pos + joint_vel +
            list(base_pos) + list(lin_vel) +
            list(base_euler) +
            contacts +                       # 4 flags: indices 33‑36
            list(self._last_action),         # 12 positions: indices 37‑48
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------ #
    #  Reward and termination                                            #
    # ------------------------------------------------------------------ #
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Reward with forward progress, stability, milestones و غیره."""
        vx, vy, vz = obs[27:30]
        pos_x, pos_y, pos_z = obs[24:27]

        # --- forward & acceleration rewards ---
        fwd_rew = 2 * vx
        drift_pen = 0.2 * (abs(vy) + abs(vz))
        second_pen = 0.4 * abs(pos_y) + 0.2 * abs(pos_z - 0.4)

        reward = fwd_rew - drift_pen - second_pen
        return float(reward)

    def _check_termination(self, obs: np.ndarray) -> bool:
        return bool(obs[26] < 0.15 or obs[26] > 0.6)
