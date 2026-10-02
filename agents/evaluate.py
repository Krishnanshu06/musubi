"""
Evaluation and visualization script for trained Musubi locomotion policies.
Automatically detects creature genome DNA from checkpoint directory to match neural network dimensions.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from envs.locomotion_env import LocomotionEnv
from morphology.genome import CreatureGenome


def resolve_model_and_genome(target_path: str) -> Tuple[str, Optional[CreatureGenome]]:
    """Resolves the .zip model path and matching creature genome."""
    if os.path.isdir(target_path):
        # Look for best_model.zip or final_model.zip
        for candidate in ["best_model.zip", "final_model.zip", "ppo_quadruped_final.zip"]:
            full_c = os.path.join(target_path, candidate)
            if os.path.exists(full_c):
                model_file = full_c
                break
        else:
            raise FileNotFoundError(f"No .zip model found inside directory: {target_path}")
        model_dir = target_path
    else:
        model_file = target_path
        model_dir = os.path.dirname(target_path)

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Could not find model file at: {model_file}")

    # Check for genome.json
    genome_file = os.path.join(model_dir, "genome.json")
    if os.path.exists(genome_file):
        genome = CreatureGenome.from_json(genome_file)
    else:
        print("⚠️ No genome.json found in model directory, defaulting to Quadruped.")
        genome = CreatureGenome.preset_quadruped()

    return model_file, genome


def evaluate_policy(
    model_path: str = "models/quadruped_runner/best_model.zip",
    num_episodes: int = 5,
    max_steps: int = 1000,
    deterministic: bool = True,
):
    """Renders the trained creature moving in the PyBullet 3D viewport."""
    model_file, genome = resolve_model_and_genome(model_path)

    print("=" * 65)
    print(" 🎮 Musubi Locomotion Policy Evaluation")
    print("=" * 65)
    print(f"Loading Model From   : {os.path.abspath(model_file)}")
    print(f"Creature Morphology  : {genome.name} ({genome.num_legs} legs, {genome.total_actuators} joints)")
    print(f"Evaluation Episodes  : {num_episodes}")
    print(f"Deterministic Policy : {deterministic}")
    print("=" * 65)

    # 1. Load the trained policy
    model = PPO.load(model_file)

    # 2. Instantiate environment with the EXACT matching creature DNA
    env = LocomotionEnv(
        render_mode="human",
        max_episode_steps=max_steps,
        creature_builder_fn=genome.build,
    )

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
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Maintain realistic ~60 FPS visual speed
                time.sleep(1.0 / 60.0)

                if terminated or truncated:
                    final_pos, _, _, _ = env.creature.get_base_state()
                    distance_traveled = final_pos[0] - start_x
                    avg_speed = distance_traveled / (step * env.frame_skip * env.time_step + 1e-6)

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

        print("\n" + "=" * 65)
        print(" Evaluation Summary Across All Episodes")
        print("=" * 65)
        print(f"Mean Reward   : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean Distance : {np.mean(episode_distances):.2f} m ± {np.std(episode_distances):.2f} m")
        print(f"Mean Speed    : {np.mean(episode_speeds):.2f} m/s ± {np.std(episode_speeds):.2f} m/s")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Musubi locomotion policy in GUI")
    parser.add_argument("--model-path", type=str, default="models", help="Path to .zip model file or experiment directory")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes to render")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of deterministic mode")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model_path,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
    )
