"""
Evaluation and visualization script for trained Musubi locomotion policies.
Loads saved neural network weights and renders real-time rollouts in PyBullet GUI.
"""

import argparse
import os
import sys
import time
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import envs  # Registers "Musubi/Locomotion-v0" with Gymnasium


def evaluate_policy(
    model_path: str = "models/ppo_quadruped_final.zip",
    num_episodes: int = 5,
    max_steps: int = 1000,
    deterministic: bool = True,
):
    """
    Renders the trained creature moving in the PyBullet 3D viewport.
    """
    if not os.path.exists(model_path):
        # Check if best_model.zip exists as fallback
        alt_path = os.path.join(os.path.dirname(model_path), "best_model.zip")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find model file at: {model_path}. Train a policy first using agents/train_sb3.py"
            )

    print("=" * 60)
    print(" Musubi Locomotion Policy Evaluation")
    print("=" * 60)
    print(f"Loading Model From   : {os.path.abspath(model_path)}")
    print(f"Evaluation Episodes  : {num_episodes}")
    print(f"Deterministic Policy : {deterministic}")
    print("=" * 60)

    # 1. Load the trained policy
    model = PPO.load(model_path)

    # 2. Instantiate environment in 3D GUI mode
    env = gym.make("Musubi/Locomotion-v0", render_mode="human", max_episode_steps=max_steps)

    episode_rewards = []
    episode_distances = []
    episode_speeds = []

    try:
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep * 42)
            total_reward = 0.0
            start_x = info.get("base_position", [0, 0, 0])[0]

            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

            for step in range(max_steps):
                # Neural network predicts continuous joint actions from observation vector
                action, _states = model.predict(obs, deterministic=deterministic)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Maintain realistic ~60 FPS visual speed
                time.sleep(1.0 / 60.0)

                if terminated or truncated:
                    current_x = info.get("base_position", [0, 0, 0])[0] if "base_position" in info else obs[0]
                    # Read ground distance traveled from creature state
                    final_pos, _, _, _ = env.unwrapped.creature.get_base_state()
                    distance_traveled = final_pos[0] - start_x
                    avg_speed = distance_traveled / (step * env.unwrapped.frame_skip * env.unwrapped.time_step + 1e-6)

                    episode_rewards.append(total_reward)
                    episode_distances.append(distance_traveled)
                    episode_speeds.append(avg_speed)

                    reason = "Terminated (Fell / Tilted)" if terminated else "Truncated (Time Limit Reached)"
                    print(f"Episode {ep + 1} Ended: {reason}")
                    print(f"  • Steps Survived    : {step + 1}")
                    print(f"  • Total Reward      : {total_reward:+.2f}")
                    print(f"  • Distance Traveled : {distance_traveled:+.2f} meters")
                    print(f"  • Avg Forward Speed : {avg_speed:+.2f} m/s")
                    break

        print("\n" + "=" * 60)
        print(" Evaluation Summary Across All Episodes")
        print("=" * 60)
        print(f"Mean Reward   : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean Distance : {np.mean(episode_distances):.2f} m ± {np.std(episode_distances):.2f} m")
        print(f"Mean Speed    : {np.mean(episode_speeds):.2f} m/s ± {np.std(episode_speeds):.2f} m/s")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Musubi locomotion policy in GUI")
    parser.add_argument("--model-path", type=str, default="models/ppo_quadruped_final.zip", help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes to render")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of deterministic mode")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model_path,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
    )
