
---

# **TEKO Vision-Based Docking System**

**Master’s Thesis Project — University of Hohenheim**
Artificial Intelligence in Agricultural Engineering

**Title:**
**Vision-Based Autonomous Docking for Modular Agricultural Robot Swarms:
Curriculum-Driven Deep Reinforcement Learning with Evolutionary Optimization**

**Author:** Alexandre Schleier Neves da Silva
**Supervisors:** Prof. Dr. Anthony Stein, Dr. David Reiser

---

## **1. Project Overview**

This repository contains the full implementation of a **vision-only autonomous docking system** for the TEKO modular agricultural robot. The main objective is to train a small mobile robot to:

1. Detect the static TEKO goal robot.
2. Align its rear connector to the goal.
3. Approach the target safely.
4. Perform a precise mechanical docking action (<3 cm error).

The entire perception–control pipeline is learned via **reinforcement learning (PPO)** inside **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0**, with no handcrafted docking logic.

---

## **2. Scientific Motivation**

Agriculture is undergoing structural changes driven by:

* Labour shortages
* Climate instability
* Pressure to reduce environmental impact
* The need for flexible, scalable mechanisation

Large monolithic machines dominate modern agriculture but are costly and inflexible. Modular swarms of small robots offer:

* Lower costs
* Redundancy
* Adaptability
* Cooperative operation

A key technological requirement is the ability of robots to **dock physically**.
This project focuses exclusively on learning the docking behaviour using **vision and PPO**.

---

## **3. System Architecture**

### **3.1 Simulation**

The simulation uses **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0** with:

* Physically accurate differential-drive TEKO robot
* CAD-based geometry converted to URDF/USD
* A static goal TEKO robot with docking connector
* A controlled docking arena

### **3.2 Robot Components**

The mobile TEKO robot includes:

* Four differential-drive wheels
* Rear RGB camera (15 Hz)
* Docking connector
* Internal spheres for geometric distance measurement
* Realistic collision meshes and wheel physics

### **3.3 Goal Robot**

Contains:

* Docking connector
* ArUco marker (not used during RL training)
* Connector spheres for success detection

### **3.4 Ground-Truth Geometry**

Docking is defined using **two virtual spheres** placed inside the connectors.

Measured quantities:

* 3D connector distance
* XY-plane projected distance (`surface_xy`)

These are used for reward shaping and success detection.

---

## **4. Observations**

The policy receives only **vision**:

* Input resolution: **84 × 84 grayscale**
* Frame stacking: **4 consecutive frames**
* Final observation shape: **[4, 84, 84]**
* Normalised to `[0, 1]`

This gives the policy temporal awareness of motion and turning direction.

---

## **5. Action Space**

The agent outputs a **2D continuous action**:

* `v` – forward/backward command in [-1, 1]
* `w` – turning command in [-1, 1]

Inside the environment, these are converted into wheel torques for the differential-drive system.

---

## **6. Visual Encoder (CNN)**

The vision encoder is a custom **IMPALA-style CNN** optimised for PPO:

```
Conv1: 32 filters, 8×8 kernel, stride 4
Conv2: 64 filters, 4×4 kernel, stride 2
Conv3: 64 filters, 3×3 kernel, stride 1
Flatten
FC1: 512 units, ReLU
FC2: 256 units, ReLU
```

Main characteristics:

* Strong large-scale receptive field
* Stable under 60 parallel environments
* Good feature extraction for offset docking and 180° turns
* Lightweight enough for RTX 3090

---

## **7. Reward Function (v8.9)**

The reward is a combination of shaping terms:

* Distance reward
* Progress reward
* Alignment reward
* Facing bonus
* Approach bonus
* Turning bonus (new)
* Collision penalty
* Boundary penalty
* Success bonus
* Time penalty

Success is detected when:

```
surface_xy < 0.03 m for ≥ 5 steps
```

Rewards are clamped to `[-500, 500]`.

---

## **8. 28-Stage Ultra-Micro Curriculum (v8.0)**

The curriculum gradually increases difficulty.
Key rule: **never increase yaw and lateral offset simultaneously**.

### **Forward Stages (S0–S3)**

Controlled distance-only docking.

### **Offset Stages (S4–S12)**

Gradual increases in yaw and lateral displacement.

### **Ultra Micro-Steps (S13–S22)**

±1 cm or ±2° increments only.

### **Large-Angle Stages (S23–S27)**

45° → 60° → 90° → 135° → 180° turn-around conditions.

### **Stage Advancement**

A stage is advanced when:

1. Minimum steps reached (50,000)
2. Success-rate threshold met
3. Or safety valve exceeded

Stage transitions include **mixing** between old and new stages to prevent forgetting.

---

## **9. PPO Implementation**

The training loop is implemented in pure PyTorch.

Key parameters:

* PPO clip ratio: 0.15
* Advantage estimation: GAE(λ = 0.95, γ = 0.99)
* Rollout length: 64 steps
* Mini-batch size: 64
* 4 PPO epochs
* Entropy coefficient adapted per stage
* Gradient clipping: 0.5

Runs comfortably with **60 environments** on an RTX 3090.

TensorBoard logs include:

* Rewards
* Stage success rate
* PPO losses
* Entropy
* Value function behaviour
* Reward components breakdown

---

## **10. Training**

### **Launching Training**

```
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/train_teko_PPO.py \
  --num_envs 60 \
  --steps 100000000 \
  --headless
```

### **Checkpoints**

Saved every 30k steps:

```
teko_curriculum/YYYYMMDD_HHMMSS/ckpt_step.pt
```

### **Curriculum State Saved in Checkpoint**

* Policy weights
* Optimiser state
* Current curriculum stage
* Steps spent in stage
* List of mastered stages

---

## **11. Repository Structure**

```
├── scripts/
│   ├── train_teko_PPO.py
│   └── evaluation scripts
├── teko/
│   ├── tasks/
│   │   ├── direct/teko_env.py
│   │   ├── rewards/reward_functions.py
│   │   ├── curriculum/curriculum_manager.py
│   │   └── teko_brain/cnn_model.py
│   ├── robots/
│   │   └── URDF + USD assets
│   └── utils/
├── documents/
│   ├── CAD/
│   ├── Images/
│   └── Thesis Figures/
└── runs/
    └── TensorBoard logs
```

---

## **12. Future Work**

* Evolutionary hyperparameter optimisation
* Real-robot training and data collection
* Domain randomisation for sim-to-real transfer
* Multi-robot cooperative docking
* Vision-only arena search without curriculum

---

## **13. License**

BSD-3-Clause License (same as Isaac Lab and your codebase).

---

## **14. Citation**

If you use this work, please cite:

```
Neves da Silva, A. S. (2025). Vision-Based Autonomous Docking for Modular
Agricultural Robot Swarms: Curriculum-Driven Deep Reinforcement Learning
with Evolutionary Optimization. University of Hohenheim.
```

---
