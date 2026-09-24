"""
Verification script to test the Musubi Locomotion Gymnasium Environment.
Tests:
1. Gymnasium environment API compliance (via check_env).
2. Headless rollouts speed test.
3. Episode reset, step, and reward signals.
"""

import time
import numpy as np
from gymnasium.utils.env_checker import check_env
from musubi.envs.locomotion_env import LocomotionEnv


def test_gymnasium_compliance():
    print("=" * 60)
    print("1. Running Gymnasium Environment API Compliance Check...")
    print("=" * 60)
    env = LocomotionEnv(render_mode=None)
    try:
        check_env(env)
        print(" [PASSED] LocomotionEnv adheres strictly to Gymnasium API standards!")
    finally:
        env.close()


def test_headless_throughput():
    print("\n" + "=" * 60)
    print("2. Running Headless Simulation Throughput Benchmark...")
    print("=" * 60)
    env = LocomotionEnv(render_mode=None)
    obs, info = env.reset(seed=42)

    num_steps = 1000
    start_time = time.time()

    for _ in range(num_steps):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_time
    sps = num_steps / elapsed
    physics_sps = sps * env.frame_skip

    print(f"Executed {num_steps} RL steps in {elapsed:.3f}s")
    print(f"Policy Decision Rate : {sps:.1f} steps/sec (Single CPU Core)")
    print(f"Physics Rate (x{env.frame_skip}) : {physics_sps:.1f} physics steps/sec")
    env.close()


def test_step_and_rewards():
    print("\n" + "=" * 60)
    print("3. Inspecting Sample Action Rollout & Rewards...")
    print("=" * 60)
    env = LocomotionEnv(render_mode=None)
    obs, info = env.reset(seed=0)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space     : {env.action_space}")
    print(f"Initial Obs Shape: {obs.shape}")

    for step_i in range(5):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_i + 1}: Reward={reward:+.4f} | FwdVel={info['forward_velocity']:+.3f} | Height={info['height']:.3f}")

    env.close()
    print("\n All foundational tests completed successfully!")


if __name__ == "__main__":
    test_gymnasium_compliance()
    test_headless_throughput()
    test_step_and_rewards()
