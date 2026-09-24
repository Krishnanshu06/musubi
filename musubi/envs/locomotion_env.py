"""
Locomotion Gymnasium environment for training creatures to walk forward.
"""

from typing import Any, Callable, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from musubi.envs.base_env import MusubiBaseEnv
from musubi.morphology.builder import Creature, MorphologyBuilder


class LocomotionEnv(MusubiBaseEnv):
    """
    Continuous control locomotion environment.
    The goal is for the creature to learn forward locomotion while remaining stable and energy-efficient.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_step: float = 1.0 / 240.0,
        frame_skip: int = 4,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        max_episode_steps: int = 1000,
        creature_builder_fn: Optional[Callable[[int], Creature]] = None,
        # Reward coefficients
        forward_reward_weight: float = 2.0,
        survival_reward_weight: float = 0.5,
        energy_penalty_weight: float = 0.005,
        tilt_penalty_weight: float = 0.1,
    ):
        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            frame_skip=frame_skip,
            gravity=gravity,
        )
        self.max_episode_steps = max_episode_steps
        self.creature_builder_fn = creature_builder_fn or MorphologyBuilder.create_quadruped

        self.forward_reward_weight = forward_reward_weight
        self.survival_reward_weight = survival_reward_weight
        self.energy_penalty_weight = energy_penalty_weight
        self.tilt_penalty_weight = tilt_penalty_weight

        self.current_step: int = 0
        self.creature: Optional[Creature] = None

        # Build prototype to inspect dimensions
        self._setup_world()
        self.creature = self.creature_builder_fn(self.client_id)
        num_actuators = self.creature.num_actuators

        # Actions: Target normalized positions [-1.0, 1.0] for all actuated joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float32
        )

        # Observation size:
        # - Torso Height: 1
        # - Torso Roll, Pitch, Yaw: 3
        # - Torso Linear Velocity (vx, vy, vz): 3
        # - Torso Angular Velocity (wx, wy, wz): 3
        # - Joint Positions: num_actuators
        # - Joint Velocities: num_actuators
        obs_dim = 1 + 3 + 3 + 3 + (2 * num_actuators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation vector for the agent."""
        pos, orn_quat, lin_vel, ang_vel = self.creature.get_base_state()
        rpy = p.getEulerFromQuaternion(orn_quat)

        joint_pos, joint_vel = self.creature.get_joint_states()

        obs = np.concatenate([
            [pos[2]],            # Height (Z)
            rpy,                 # Roll, Pitch, Yaw
            lin_vel,             # Linear velocity
            ang_vel,             # Angular velocity
            joint_pos,           # Joint angles
            joint_vel,           # Joint speeds
        ]).astype(np.float32)

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        # Spawn fresh creature
        self.creature = self.creature_builder_fn(self.client_id)

        # Let it settle slightly on ground
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_observation()
        info = {
            "num_actuators": self.creature.num_actuators,
            "base_position": self.creature.get_base_state()[0],
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply motor action and step physics
        self.creature.apply_action_positions(action)
        self.step_simulation()

        obs = self._get_observation()
        pos, orn_quat, lin_vel, ang_vel = self.creature.get_base_state()
        roll, pitch, yaw = p.getEulerFromQuaternion(orn_quat)

        # Compute multi-objective reward
        forward_vel = float(lin_vel[0])
        r_forward = self.forward_reward_weight * forward_vel
        r_survival = self.survival_reward_weight
        p_energy = self.energy_penalty_weight * float(np.sum(np.square(action)))
        p_tilt = self.tilt_penalty_weight * (abs(roll) + abs(pitch))

        reward = r_forward + r_survival - p_energy - p_tilt

        # Check termination (falling over or flipping upside down)
        is_flipped = abs(roll) > (np.pi / 2) or abs(pitch) > (np.pi / 2)
        is_too_low = pos[2] < 0.12

        terminated = bool(is_flipped or is_too_low)
        truncated = bool(self.current_step >= self.max_episode_steps)

        if self.render_mode == "human":
            self.update_camera_tracking(pos)

        info = {
            "forward_velocity": forward_vel,
            "height": float(pos[2]),
            "reward_forward": r_forward,
            "penalty_energy": p_energy,
            "penalty_tilt": p_tilt,
        }

        return obs, float(reward), terminated, truncated, info
