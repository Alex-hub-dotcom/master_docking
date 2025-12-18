# TEKO Vision-Based Autonomous Docking System

<p align="center">
  <img src="docs/images/teko_docking.png" alt="TEKO Docking" width="600"/>
</p>

**Master's Thesis Project — University of Hohenheim**  
*Artificial Intelligence in Agricultural Engineering*

**Author:** Alexandre Schleier Neves da Silva  
**Supervisors:** Prof. Dr. Anthony Stein, Dr. David Reiser  
**Date:** December 2024

---

## Table of Contents

- [Overview](#overview)
- [Scientific Motivation](#scientific-motivation)
- [System Architecture](#system-architecture)
- [Reinforcement Learning Pipeline](#reinforcement-learning-pipeline)
- [Curriculum Learning](#curriculum-learning)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

---

## Overview

This repository contains the full implementation of a **vision-only autonomous docking system** for the TEKO modular agricultural robot. The system trains a small mobile robot to:

1. **Detect** the static TEKO goal robot using only an onboard camera
2. **Align** its rear connector with the goal's docking interface
3. **Approach** the target safely while maintaining alignment
4. **Dock** with precision (<3 cm error)

The entire perception-control pipeline is learned via **Proximal Policy Optimization (PPO)** inside **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0**, with no handcrafted docking logic.

### Key Features

- **Pure Vision Input**: 84×84 grayscale images with 4-frame stacking
- **End-to-End Learning**: From pixels to wheel torques
- **28-Stage Curriculum**: Gradual difficulty progression from easy to 180° turns
- **Asymmetric Actor-Critic**: Vision-only actor, vision+privileged critic
- **Distributed Training**: Up to 256 parallel environments on RTX 3090
- **Hyperparameter Optimization**: Optuna-based distributed HPO with pruning

---

## Scientific Motivation

Modern agriculture faces critical challenges:

| Challenge | Impact |
|-----------|--------|
| Labour shortages | Reduced workforce availability |
| Climate instability | Unpredictable growing conditions |
| Environmental pressure | Need for precision application |
| Cost of large machinery | High capital requirements |

**Modular robot swarms** offer a promising alternative to monolithic machines:

- ✅ Lower individual cost
- ✅ Redundancy and fault tolerance
- ✅ Scalable to field size
- ✅ Cooperative task execution

A fundamental capability for swarm operation is **autonomous docking** — enabling robots to physically connect for:
- Energy transfer
- Tool sharing
- Coordinated transport
- Collective manipulation

This project develops a **learned docking behavior** using only onboard vision, eliminating the need for external positioning systems or fiducial markers during operation.

---

## System Architecture

### Simulation Environment

Built on **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab Environment                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Active TEKO │    │  Goal TEKO  │    │   Arena Stage   │  │
│  │   Robot     │───▶│   (Static)  │    │                 │  │
│  │             │    │             │    │  2.4m × 3.6m    │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │ Rear Camera │    │  Connector  │                         │
│  │  84×84 RGB  │    │   Spheres   │                         │
│  └─────────────┘    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Robot Configuration

| Component | Specification |
|-----------|---------------|
| Drive system | 4-wheel differential drive |
| Camera | Rear-mounted RGB, 84×84 @ 15 Hz |
| Connector | Female rear connector with alignment sphere |
| Dimensions | ~35 cm × 20 cm footprint |
| Actuation | Torque-controlled wheels (±1.2 Nm) |

### Docking Geometry

Success is measured using **virtual spheres** inside the connectors:

```
Active Robot                    Goal Robot
    ┌───┐                          ┌───┐
    │   │◄── Female Sphere         │   │◄── Male Sphere
    │   │    (r=5mm)               │   │    (r=5mm)
    └───┘                          └───┘
    
Success Criterion: surface_xy < 30mm for ≥5 consecutive steps
```

---

## Reinforcement Learning Pipeline

### Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| RGB frames | `[4, 84, 84]` | Grayscale frame stack, normalized to [0,1] |
| Privileged state* | `[7]` | dx, dy, dz, yaw_error, vx, vy, ω |

*Privileged state used only by critic during training (asymmetric architecture)

### Action Space

| Action | Range | Description |
|--------|-------|-------------|
| `v` | [-1, 1] | Forward/backward velocity command |
| `ω` | [-1, 1] | Angular velocity command |

Converted to differential wheel torques internally.

### Neural Network Architecture

#### Vision Encoder (SimpleCNN)

```
Input: [B, 4, 84, 84]
    │
    ▼
Conv2D(4→32, k=6, s=3, p=1) + LayerNorm + ReLU  →  [B, 32, 27, 27]
    │
    ▼
Conv2D(32→64, k=4, s=2, p=1) + LayerNorm + ReLU  →  [B, 64, 13, 13]
    │
    ▼
Conv2D(64→128, k=3, s=2, p=1) + LayerNorm + ReLU  →  [B, 128, 7, 7]
    │
    ▼
Flatten  →  [B, 6272]
    │
    ▼
Linear(6272→512) + ReLU  →  [B, 512]
    │
    ▼
Linear(512→256) + ReLU  →  [B, 256]
```

#### Asymmetric Actor-Critic

```
                    ┌──────────────────┐
                    │  Vision Encoder  │
                    │    (Shared)      │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
       ┌─────────────┐               ┌─────────────────┐
       │    Actor    │               │ State Encoder   │
       │ (256→128→64)│               │   (7→128→128)   │
       └──────┬──────┘               └────────┬────────┘
              │                               │
              ▼                               ▼
       ┌─────────────┐               ┌─────────────────┐
       │   Action    │               │     Critic      │
       │  mean + std │               │  (384→128→64→1) │
       └─────────────┘               └─────────────────┘
       
Actor: Vision only (deployment-ready)
Critic: Vision + Privileged (faster training)
```

### Reward Function (v9.1)

| Component | Scale | Description |
|-----------|-------|-------------|
| Distance | -2.0 | Continuous penalty for distance |
| Progress | 8.0 | Reward for getting closer |
| Alignment | 0.20 | Reward for correct orientation |
| Facing bonus | 1.0 | Bonus when aligned and close |
| Approach bonus | 2.0 | Bonus for approaching while aligned |
| Turning bonus | 0.35 | Bonus for correcting misalignment |
| Collision penalty | -100 | Terminal penalty for crashes |
| Boundary penalty | -500 | Terminal penalty for leaving arena |
| Success bonus | +400 | Terminal reward for successful docking |
| Time penalty | Exponential | Encourages fast completion |

---

## Curriculum Learning

### 28-Stage Curriculum (v9.1)

The curriculum follows a key principle: **never increase yaw AND lateral offset simultaneously**.

#### Stage Progression

| Stages | Type | Yaw Range | Lateral Range | Distance |
|--------|------|-----------|---------------|----------|
| S0-S3 | Forward | 0° | 0 cm | 5-35 cm |
| S4-S6 | First Offsets | ±4° to ±10° | ±2-3 cm | 25-38 cm |
| S7-S12 | Micro-steps | ±11° to ±19° | ±3-4 cm | 25-38 cm |
| S13-S22 | Ultra micro | ±20° to ±30° | ±4-10 cm | 25-40 cm |
| S23-S27 | Large angles | ±45° to ±180° | ±3-8 cm | 28-54 cm |

#### Stage Advancement Criteria

```python
advancement_conditions = {
    "min_steps": 50_000,           # Minimum training in stage
    "max_steps": 1_000_000,        # Safety valve
    "ssr_threshold": {             # Stage Success Rate
        "S0-S4":  0.75,
        "S5-S6":  0.70,
        "S7-S12": 0.65,
        "S13-S22": 0.60,
        "S23-S27": 0.55,
    }
}
```

#### Anti-Forgetting Mechanisms

1. **Stage Mixing**: When advancing, train on both current and next stage
2. **Rehearsal**: Periodically replay on previously mastered stages
3. **Replay Probability**: 15-25% of resets spawn in previous stage

---

## Hyperparameter Optimization

### Optuna Configuration

Distributed HPO using SQLite storage for multi-worker coordination.

#### Search Space

| Hyperparameter | Range | Scale |
|----------------|-------|-------|
| `entropy_coef` | [0.0, 0.01] | Linear |
| `gae_lambda` | [0.9, 1.0] | Linear |
| `clip_ratio` | {0.1, 0.2, 0.3} | Categorical |
| `epochs` | [3, 30] | Integer |
| `learning_rate` | [1e-5, 1e-3] | Log |
| `batch_size` | {1024, 2048, 4096} | Categorical |

#### Fixed Parameters

| Parameter | Value |
|-----------|-------|
| `gamma` | 0.99 |
| `value_clip` | 0.2 |
| `value_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `rollout_len` | 128 |
| `num_envs` | 150 |

#### Pruning Strategy

Trials are pruned early if they show poor performance:

| Checkpoint | Condition | Action |
|------------|-----------|--------|
| 100k steps | SSR < 30% at S0 | Prune |
| 200k steps | Stuck at S≤3 | Prune |
| 500k steps | Stuck at S≤5 | Prune |

### Running HPO

```bash
# Create study (once)
sbatch scripts/optuna/create_optuna_study.sh

# Launch workers (7 GPUs)
./scripts/optuna/launch_optuna_workers.sh 7

# Monitor progress
python scripts/optuna/check_optuna_progress.py
```

---

## Installation

### Requirements

- NVIDIA GPU (RTX 3090 recommended, 24GB VRAM)
- NVIDIA Driver 550+
- Docker or Enroot container runtime
- Isaac Sim 5.0 / Isaac Lab 0.47.1

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/teko.git
cd teko

# Pull Isaac Lab container
enroot import docker://nvcr.io/nvidia/isaac-sim:5.0.0

# Create workspace
mkdir -p /workspace/teko
cp -r . /workspace/teko/

# Install Python dependencies (inside container)
pip install optuna tensorboard
```

---

## Usage

### Training

```bash
# Single training run
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/skrl/train_ppo_final.py \
    --num_envs 150 \
    --steps 100000000 \
    --headless

# With checkpoint resume
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/skrl/train_ppo_final.py \
    --checkpoint experiments/run_20241218/ckpt_1000000.pt \
    --headless
```

### Evaluation

```bash
# Evaluate trained policy
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/evaluate_policy.py \
    --checkpoint experiments/best_model.pt \
    --num_episodes 100
```

### Visualization

```bash
# Run with rendering (requires display)
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/skrl/train_ppo_final.py \
    --num_envs 4
```

---

## Repository Structure

```
teko/
├── documents/
│   ├── CAD/
│   │   └── USD/                    # Robot and arena USD files
│   ├── Aruco/                      # ArUco marker textures
│   └── Images/                     # Documentation images
│
├── source/teko/teko/
│   └── tasks/direct/teko/
│       ├── teko_env.py             # Main environment
│       ├── teko_env_cfg.py         # Environment configuration
│       ├── rewards/
│       │   └── reward_functions.py # Reward computation (v9.1)
│       ├── curriculum/
│       │   └── curriculum_manager.py # 28-stage curriculum
│       ├── robots/
│       │   ├── teko.py             # Active robot config
│       │   └── teko_static.py      # Goal robot spawner
│       └── utils/
│           ├── geometry_utils.py   # Quaternion/pose utilities
│           └── logging_utils.py    # TensorBoard helpers
│
├── scripts/
│   ├── skrl/
│   │   └── train_ppo_final.py      # Main training script
│   ├── optuna/
│   │   ├── train_optuna_ppo.py     # HPO training script
│   │   ├── run_optuna_worker.sh    # SLURM worker script
│   │   ├── create_optuna_study.sh  # Study initialization
│   │   ├── launch_optuna_workers.sh # Multi-worker launcher
│   │   └── check_optuna_progress.py # Progress monitor
│   └── sanity_checks/
│       └── debug_spawn_goal_overlap.py # Docking validation
│
├── running_scripts/
│   └── sanity/
│       └── run_sanity_goal.sh      # SLURM sanity check
│
├── experiments/                     # Training outputs
│   └── ppo_final/
│       └── YYYYMMDD_HHMMSS/
│           ├── ckpt_*.pt           # Checkpoints
│           └── events.*            # TensorBoard logs
│
└── optuna/
    └── teko_study.db               # Optuna SQLite database
```

---

## Results

### Training Progress

*Results pending completion of Optuna optimization...*

| Metric | Value |
|--------|-------|
| Maximum stage reached | TBD |
| Best SSR | TBD |
| Training time | TBD |
| Best hyperparameters | TBD |

### Docking Success Rate by Stage

*To be updated after training completion*

---

## Technical Notes

### Coordinate Systems

- **Robot Local Frame**: +X forward, -X rear (URDF convention)
- **World Frame**: Standard ROS convention
- **Quaternion Format**: (w, x, y, z) — Isaac Lab convention

### Key Bug Fixes

1. **Female Sphere Offset**: Changed from `[0.24, 0.0, -0.08]` to `[-0.24, 0.0, -0.08]` to account for robot's 180° spawn rotation
2. **SQLite Permissions**: Added `:rw` flag to Enroot mount for Optuna database writes

---

## License

BSD-3-Clause License

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{nevesilva2025teko,
    author = {Neves da Silva, Alexandre Schleier},
    title = {Vision-Based Autonomous Docking for Modular Agricultural Robot Swarms: 
             Curriculum-Driven Deep Reinforcement Learning with Evolutionary Optimization},
    school = {University of Hohenheim},
    year = {2025},
    type = {Master's Thesis},
    address = {Stuttgart, Germany}
}
```

---

## Acknowledgments

- **Prof. Dr. Anthony Stein** — Thesis supervision and guidance
- **Dr. David Reiser** — Technical supervision and agricultural robotics expertise
- **University of Hohenheim** — Computational resources and support
- **NVIDIA** — Isaac Sim/Lab platform

---

<p align="center">
  <i>Developed at the University of Hohenheim, Institute for Agricultural Engineering</i>
</p>