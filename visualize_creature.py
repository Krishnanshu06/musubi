"""
Interactive visual test for the Musubi Locomotion Environment.
Runs the simulation in PyBullet GUI mode with an open-loop sinusoidal gait and dynamic camera tracking.
"""

import time
import numpy as np
import musubi  # noqa: F401
from musubi.envs.locomotion_env import LocomotionEnv


def run_visual_demo(episodes: int = 3, max_steps: int = 400):
    print("Launching Musubi in GUI mode...")
    env = LocomotionEnv(render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        print(f"\n--- Episode {ep + 1}/{episodes} (Actuators: {info['num_actuators']}) ---")
        total_reward = 0.0

        for step in range(max_steps):
            # Generate a simple oscillating wave pattern across joints to test motor responses
            t = step * 0.1
            action = np.array([
                np.sin(t),           # FL Hip
                np.sin(t + np.pi/2), # FL Knee
                np.cos(t),           # FR Hip
                np.cos(t + np.pi/2), # FR Knee
                np.sin(t + np.pi),   # BL Hip
                np.sin(t + 3*np.pi/2), # BL Knee
                np.cos(t + np.pi),   # BR Hip
                np.cos(t + 3*np.pi/2)  # BR Knee
            ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Maintain ~60 FPS visual pace for human observation
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                print(f"Episode ended at step {step + 1} | Total Reward: {total_reward:.2f}")
                break

    env.close()
    print("Visual demo completed!")


if __name__ == "__main__":
    run_visual_demo()
