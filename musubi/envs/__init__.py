"""
Environments module for Musubi.
Registers Musubi environments with Gymnasium.
"""

import gymnasium as gym
from musubi.envs.base_env import MusubiBaseEnv
from musubi.envs.locomotion_env import LocomotionEnv

# Register the locomotion environment for easy gym.make() instantiation
gym.register(
    id="Musubi/Locomotion-v0",
    entry_point="musubi.envs.locomotion_env:LocomotionEnv",
    max_episode_steps=1000,
)

__all__ = ["MusubiBaseEnv", "LocomotionEnv"]
