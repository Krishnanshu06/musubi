"""
Training script for continuous locomotion policies using Stable-Baselines3 (PPO).
Supports creature-aware checkpoints, saving genome.json DNA alongside model weights.
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.locomotion_env import LocomotionEnv
from morphology.genome import CreatureGenome


def resolve_genome(creature_arg: str) -> CreatureGenome:
    """Resolves creature genome from preset name or JSON file path."""
    arg_lower = creature_arg.lower()
    if arg_lower == "hexapod":
        return CreatureGenome.preset_hexapod()
    elif arg_lower == "biped":
        return CreatureGenome.preset_biped()
    elif arg_lower in ["centipede", "crawler"]:
        return CreatureGenome.random(num_pairs=4, segments_per_leg=2)
    elif arg_lower == "quadruped":
        return CreatureGenome.preset_quadruped()
    elif os.path.exists(creature_arg):
        return CreatureGenome.from_json(creature_arg)
    else:
        print(f"Unknown preset '{creature_arg}', defaulting to Quadruped.")
        return CreatureGenome.preset_quadruped()


def make_env(genome: CreatureGenome, rank: int = 0, seed: int = 0) -> Callable[[], gym.Env]:
    """Utility helper to create and seed single isolated environment instances with matching creature DNA."""
    def _init() -> gym.Env:
        env = LocomotionEnv(render_mode=None, creature_builder_fn=genome.build)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(
    creature_name: str = "quadruped",
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    save_base_dir: str = "models",
    log_base_dir: str = "runs",
    seed: int = 42,
):
    """Executes creature-aware PPO training."""
    # 1. Resolve Creature Genome
    genome = resolve_genome(creature_name)
    exp_name = genome.name.lower().replace(" ", "_")

    exp_model_dir = os.path.join(save_base_dir, exp_name)
    exp_log_dir = os.path.join(log_base_dir, exp_name)

    os.makedirs(exp_model_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    # 2. Save Creature DNA metadata in the experiment folder
    genome_file = os.path.join(exp_model_dir, "genome.json")
    genome.save_json(genome_file)

    print("=" * 65)
    print(f" 🚀 Musubi RL Locomotion Training: [{genome.name}]")
    print("=" * 65)
    print(f"Creature Type          : {genome.name} ({genome.num_legs} legs, {genome.total_actuators} joints)")
    print(f"Target Total Timesteps : {total_timesteps:,}")
    print(f"Parallel Worker Envs   : {n_envs}")
    print(f"Learning Rate          : {learning_rate}")
    print(f"Model Checkpoint Dir   : {os.path.abspath(exp_model_dir)}")
    print(f"Creature DNA Saved To  : {os.path.abspath(genome_file)}")
    print(f"TensorBoard Log Dir    : {os.path.abspath(exp_log_dir)}")
    print("=" * 65)

    # 3. Create Vectorized Environments
    if n_envs > 1:
        env_fns = [make_env(genome=genome, rank=i, seed=seed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv([make_env(genome=genome, rank=0, seed=seed)])

    eval_env = DummyVecEnv([make_env(genome=genome, rank=100, seed=seed + 100)])

    # 4. Setup Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_model_dir,
        log_path=exp_log_dir,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=exp_model_dir,
        name_prefix="checkpoint_step",
    )

    # 5. Policy Architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    # 6. Instantiate PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=exp_log_dir,
        device="cpu",
        verbose=1,
        seed=seed,
    )

    # 7. Start Training
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False,
        )
    finally:
        total_time = time.time() - start_time
        print(f"\n Training completed in {total_time:.2f} seconds ({total_time / 60.0:.2f} minutes)!")

        # Save Final Model
        final_path = os.path.join(exp_model_dir, "final_model.zip")
        model.save(final_path)
        print(f" Final model saved successfully to: {final_path}")

        # Save Training Summary
        summary = {
            "creature_name": genome.name,
            "num_legs": genome.num_legs,
            "num_actuators": genome.total_actuators,
            "total_timesteps": total_timesteps,
            "elapsed_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(exp_model_dir, "training_info.json"), "w") as f:
            json.dump(summary, f, indent=2)

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Musubi Locomotion Policy with PPO")
    parser.add_argument("--creature", type=str, default="quadruped", help="Creature preset: quadruped, hexapod, biped, centipede, or path to genome.json")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total environment steps to train")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel CPU worker environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--save-dir", type=str, default="models", help="Base directory to save models")
    parser.add_argument("--log-dir", type=str, default="runs", help="Base directory to save TensorBoard logs")
    args = parser.parse_args()

    train(
        creature_name=args.creature,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        save_base_dir=args.save_dir,
        log_base_dir=args.log_dir,
    )
