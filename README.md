# Musubi 🍙

An evolutionary robotics and reinforcement learning sandbox built with **PyBullet** and **Gymnasium**.

Musubi is designed to simulate procedural 3D creatures, co-evolve body morphologies, and train locomotion controllers using continuous Reinforcement Learning (RL).

---

## 🚀 Key Features

- **Custom Gymnasium Environments**: Fully compliant with modern Gymnasium API (`reset`, `step`, `render`, `close`).
- **Decoupled Physics & Sub-stepping**: 240 Hz physics simulation coupled with 60 Hz agent decision frequency for stable, natural locomotion.
- **High-Throughput Simulation**: Headless mode executing ~2,400+ decision steps/sec per single CPU core.
- **Procedural Morphology Engine**: Programmatic creation of articulated multi-body creatures (e.g. quadrupeds, crawlers) with customizable kinematic trees.
- **Multi-Objective Reward Shaping**: Balanced forward velocity reward, upright survival bonus, energy penalty, and tilt stabilization.

---

## 🛠️ Tech Stack

- **Physics Simulation**: `PyBullet` (3.25)
- **RL Framework**: `Gymnasium` (1.2.3), `Stable-Baselines3` (2.9.0)
- **Deep Learning**: `PyTorch` (2.10) with CUDA acceleration
- **UI & Visualization**: `PySide6` / `Qt6`
- **Data & Computation**: `NumPy`, `Pandas`, `Matplotlib`, `TensorBoard`

---

## 📁 Repository Structure

```text
Musubi/
├── envs/               # Custom Gymnasium simulation environments
│   ├── __init__.py     # Gymnasium environment registration
│   ├── base_env.py     # Base PyBullet-Gymnasium wrapper
│   └── locomotion_env.py # Continuous locomotion task
├── morphology/         # Procedural multi-body creature generation
│   ├── __init__.py
│   └── builder.py      # Creature wrapper & MorphologyBuilder
├── agents/             # RL training pipelines (PPO/SAC)
├── test_env.py         # Gymnasium API compliance & throughput benchmark
├── visualize_creature.py # Interactive GUI visualizer with tracking camera
├── environment.yml     # Conda environment specifications
└── README.md
```

---

## ⚡ Quickstart

### 1. Environment Setup

Activate the conda environment:

```powershell
conda env create -f environment.yml
conda activate musubi
```

### 2. Verify Environment Compliance & Benchmark

Run the verification test:

```powershell
python test_env.py
```

### 3. Launch Interactive Visual Demo

View the creature in the PyBullet 3D visualizer:

```powershell
python visualize_creature.py
```
