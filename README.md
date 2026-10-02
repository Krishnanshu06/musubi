# Musubi 🍙

An evolutionary robotics and reinforcement learning sandbox built with **PyBullet** and **Gymnasium**.

Musubi is designed to simulate procedural 3D creatures, co-evolve body morphologies, and train locomotion controllers using continuous Reinforcement Learning (RL).

---

## 🚀 Key Features

- **Custom Gymnasium Environments**: Fully compliant with modern Gymnasium API (`reset`, `step`, `render`, `close`).
- **Decoupled Physics & Sub-stepping**: 240 Hz physics simulation coupled with 60 Hz agent decision frequency for stable, natural locomotion.
- **High-Throughput Simulation**: Headless mode executing ~2,400+ decision steps/sec per single CPU core.
- **Interactive GUI Suite (`apps/`)**: Dedicated desktop applications for designing robots (`modelBuilder.py`) and evaluating trained models (`modelEvaluate.py`).
- **URDF Studio & Creature Designer (`apps/modelBuilder.py`)**: A modern PySide6 GUI to design custom creatures, adjust joints and masses, preview in 3D, and export standard `.urdf` files.
- **Trained Model Player & Evaluation Dashboard (`apps/modelEvaluate.py`)**: Automatically discovers all trained models, matches their creature DNA, and launches real-time 3D evaluations with live telemetry.
- **Creature-Aware Model Checkpointing**: Checkpoints save matching `genome.json` DNA alongside PyTorch weights so policies always spawn their exact body morphology without dimension errors.
- **Continuous RL Training Pipeline (`agents/train_sb3.py`)**: Multi-core parallel PPO training with TensorBoard logging.

---

## 🛠️ Tech Stack

- **Physics Simulation**: `PyBullet` (3.25)
- **RL Framework**: `Gymnasium` (1.2.3), `Stable-Baselines3` (2.9.0)
- **Deep Learning**: `PyTorch` (2.10)
- **UI & Visualization**: `PySide6` / `Qt6`
- **Data & Computation**: `NumPy`, `Pandas`, `Matplotlib`, `TensorBoard`

---

## 📁 Repository Structure

```text
Musubi/
├── apps/                     # Dedicated Desktop GUI Applications
│   ├── __init__.py
│   ├── modelBuilder.py       # 🧬 URDF Studio & Creature Designer App
│   └── modelEvaluate.py      # 🎮 Trained Policy Player & Evaluation Dashboard
├── envs/                     # Custom Gymnasium simulation environments
│   ├── __init__.py           # Gymnasium environment registration
│   ├── base_env.py           # Base PyBullet-Gymnasium wrapper
│   └── locomotion_env.py     # Continuous locomotion task
├── morphology/               # Procedural multi-body creature generation
│   ├── __init__.py
│   ├── builder.py            # Creature wrapper & joint inspection
│   ├── genome.py             # DNA engine, mutation, crossover & PyBullet compiler
│   └── urdf_exporter.py      # Standard XML URDF serialization engine
├── agents/                   # RL training and evaluation pipelines
│   ├── __init__.py
│   ├── train_sb3.py          # Multi-core PPO training engine
│   └── evaluate.py           # CLI evaluation & performance benchmarking
├── showcase_creatures.py     # Interactive visual preview of procedural creatures
├── test_env.py               # Gymnasium API compliance & throughput benchmark
├── visualize_creature.py     # Interactive GUI visualizer with tracking camera
├── environment.yml           # Conda environment specifications
└── README.md
```

---

## ⚡ Quickstart

### 1. Launch the Trained Model Player
Discover all trained models and watch them walk live in 3D:
```powershell
python apps/modelEvaluate.py
```

### 2. Launch the Desktop URDF Studio & Model Builder
Design custom robots with real-time sliders, test 3D joint flex, and export `.urdf` files:
```powershell
python apps/modelBuilder.py
```

### 3. Train a Locomotion Policy for Any Creature (PPO)
Train an agent across 4 parallel CPU cores (e.g. Hexapod, Quadruped, Biped, Centipede):
```powershell
# Trains Hexapod and saves into models/hexapod_spider/
python agents/train_sb3.py --creature hexapod --timesteps 300000 --n-envs 4

# Trains Quadruped and saves into models/quadruped_runner/
python agents/train_sb3.py --creature quadruped --timesteps 300000 --n-envs 4
```

### 4. CLI Evaluation
Run evaluation directly from command line:
```powershell
python agents/evaluate.py --model-path models/hexapod_spider --episodes 3
```
