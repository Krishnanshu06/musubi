"""
Training script for training continuous locomotion policies using Stable-Baselines3 (PPO).
Supports multi-core parallel environment rollouts, TensorBoard logging, and evaluation checkpoints.
"""

import argparse
import os
import sys
import time
from typing import Callable

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import envs  # Registers "Musubi/Locomotion-v0" with Gymnasium


def make_env(env_id: str = "Musubi/Locomotion-v0", rank: int = 0, seed: int = 0) -> Callable[[], gym.Env]:
    """
    Utility helper to create and seed single isolated environment instances.
    Wrapping with Monitor records episode returns, lengths, and metrics for TensorBoard.
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=None)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    save_dir: str = "models",
    log_dir: str = "runs",
    seed: int = 42,
):
    """
    Executes the PPO training loop with vectorized parallel rollouts.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(" Musubi Reinforcement Learning Training Pipeline (PPO)")
    print("=" * 60)
    print(f"Target Total Timesteps : {total_timesteps:,}")
    print(f"Parallel Worker Envs   : {n_envs}")
    print(f"Learning Rate          : {learning_rate}")
    print(f"Mini-batch Size        : {batch_size}")
    print(f"Rollout Steps / Worker : {n_steps} (Total batch: {n_steps * n_envs})")
    print(f"Model Checkpoint Dir   : {os.path.abspath(save_dir)}")
    print(f"TensorBoard Log Dir    : {os.path.abspath(log_dir)}")
    print("=" * 60)

    # 1. Create Vectorized Environments for Parallel CPU Rollouts
    # If 1 env, use DummyVecEnv; if multiple envs, use SubprocVecEnv for true multi-core speedup
    if n_envs > 1:
        env_fns = [make_env(rank=i, seed=seed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv([make_env(rank=0, seed=seed)])

    # Separate evaluation environment to periodically test the policy deterministically
    eval_env = DummyVecEnv([make_env(rank=100, seed=seed + 100)])

    # 2. Setup Callbacks
    # Saves the 'best_model.zip' automatically whenever the agent achieves a higher mean reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Saves periodic snapshot checkpoints every 100,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="ppo_quadruped_step",
    )

    # 3. Configure Neural Network Policy Architecture
    # 2 hidden layers with 256 units each for both the Actor (Policy) and Critic (Value function)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    # 4. Instantiate PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,          # Discount factor for future rewards
        gae_lambda=0.95,     # Generalized Advantage Estimation smoothing factor
        clip_range=0.2,      # PPO surrogate objective clipping range
        ent_coef=0.0,        # Entropy coefficient for exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cpu",        # CPU is fastest for small MLPs with vectorized envs
        verbose=1,
        seed=seed,
    )

    # 5. Start Training Loop
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

        # 6. Save Final Model
        final_path = os.path.join(save_dir, "ppo_quadruped_final.zip")
        model.save(final_path)
        print(f" Final model saved successfully to: {final_path}")

        # Clean up environment processes
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Musubi Quadruped Locomotion with PPO")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total environment steps to train")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel CPU worker environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory to save TensorBoard logs")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
