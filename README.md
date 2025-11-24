# TEKO Vision-Based Docking System

This repository is part of the master's thesis project  
**"Adaptive Cooperation in Agricultural Robot Swarms: Reinforcement Learning and Evolutionary Algorithms for Modular Docking"**.

The work is being carried out at the **University of Hohenheim**,  
Department of Artificial Intelligence in Agricultural Engineering,  
under the supervision of **Prof. Dr. Anthony Stein** and **Dr. David Reiser**.

---

## 1. Project Overview

The TEKO project studies how small agricultural robots can **physically dock** to form modular units capable of performing tasks that would traditionally require larger, more complex machines.

This repository implements a **vision-only autonomous docking system**:

* A mobile TEKO robot must **locate, align, and connect** to a static TEKO goal robot.
* Perception is based purely on **RGB images** from a rear-mounted camera (no ArUco markers used during training).
* Control is learned with **reinforcement learning (PPO)** in **NVIDIA Isaac Lab 0.47.1 / Isaac Sim 5.0**.
* The setup is designed to be later transferable to **real TEKO hardware**.

The code covers the full pipeline: CAD → USD/URDF models → simulated environment → RL training → logging and analysis.

---

## 2. Research Motivation

Modern agriculture faces:

* Labour shortages and high labour costs  
* Increased climate variability and production risks  
* Pressure to reduce inputs and environmental impact  

Traditional solutions rely on **large, monolithic machines**, which are expensive and not always suitable for smaller or more diversified farms.

**Swarm and modular robotics** offer an alternative: many small, affordable units that can act **individually or cooperatively**. A key technical challenge is enabling **robust physical cooperation**, such as mechanical docking and resource sharing.

This project focuses on the **autonomous docking behaviour** itself, using vision-based RL to make two small robots **connect reliably without handcrafted docking sequences**.

---

## 3. Main Objectives

1. **Design and model** a docking-capable TEKO robot in simulation (USD/URDF, CAD-based geometry).  
2. **Implement a realistic docking arena**, including a goal robot with an ArUco marker and a well-defined docking interface.  
3. **Train a reinforcement learning agent** to perform docking using **only RGB input** from a rear camera.  
4. **Introduce a multi-stage curriculum** that gradually increases task difficulty (distance, lateral offset, orientation).  
5. **Prepare the pipeline for evolutionary hyperparameter optimisation** and later **sim-to-real transfer**.

---

## 4. System Overview

### 4.1 Simulation and Robot Models

The simulation is implemented in **NVIDIA Isaac Lab 0.47.1** (Isaac Sim 5.0):

* TEKO robot exported from **Fusion 360** as meshes and assembled into USD/URDF.
* The robot includes:
  * Chassis, four differential-drive wheels, body, roof and sensor mounts.
  * Back-mounted **camera module** emulating the Raspberry Pi Camera Module 2.
  * A **rear connector** (male/female) used for mechanical docking.
* A separate **static TEKO goal** is spawned as the docking target, equipped with:
  * An **ArUco marker** in front of the connector (visual reference, not used during RL training).
  * Spheres on both robots to define a geometric **docking distance**.

The arena (`stage_arena.usd`) defines walls and floor, constraining the robot to a controlled region.

### 4.2 Docking Geometry and Ground Truth

Docking quality is measured using **virtual spheres** placed in the connectors of both robots. The environment computes:

* **3D distance** between the connector spheres,  
* **Projected XY distance** on the ground plane (`surface_xy`),

and uses these distances for:

* **Reward shaping** (distance, progress, proximity),  
* **Success detection** (dock if `surface_xy < 0.03 m`),  
* **Collision detection** (too fast / too close → heavy penalty).  

This keeps the learning signal **geometric and consistent**, independent of camera artefacts.

---

## 5. Reinforcement Learning Setup

### 5.1 Observations

* **Modality:** RGB images from the **rear camera** of the mobile robot.  
* **Resolution:** `640 × 480` (3 channels, `float32` in `[0, 1]`).  
* **Frame Stacking:** 4 consecutive frames are stacked together, resulting in a 12-channel input `[B, 12, H, W]` to provide temporal context and motion cues.  
* **Normalization:** Images are normalized to `[0, 1]` range in the environment; the CNN processes them directly without additional normalization.  
* **Viewpoint:** The rear camera looks toward the docking interface when the robot is correctly positioned.

### 5.2 Action Space

The policy outputs a **2D continuous action vector**:

* `v_cmd` – forward/backward command (linear component) in `[-1, 1]`  
* `w_cmd` – turning command (angular component) in `[-1, 1]`  

These commands are then **mapped inside the environment** to **wheel torques** for the left and right wheel pairs using differential drive kinematics.

The `[v_cmd, w_cmd]` parameterisation:

* Encodes the **natural structure of differential drive**,  
* Makes the control space **more interpretable**,  
* Encourages **smoother and more consistent** docking behaviour,  
* Still remains a **continuous** action space.

### 5.3 Reward Function

The reward function (see `reward_functions.py` v7.4) combines multiple components to encourage safe, efficient docking:

1. **Distance reward** (`-2.5 × surface_xy`, clamped to `[-12, 0]`) – Provides gradient toward the goal at medium ranges.  
2. **Progress reward** (`12.0 × Δdistance`, clamped to `[-5, 5]`) – Main dense signal; rewards reducing distance to goal.  
3. **Alignment reward** (`3.0 × cos(yaw_error)`, range `[-3, 3]`) – **Critical for offset stages**; ensures the **rear of the robot** faces the goal.  
4. **Facing bonus** (`+3.0`) – Awarded when close (<15 cm) and well-aligned (<30°).  
5. **Approach bonus** (`2.0 × progress`) – Extra reward for closing distance while roughly aligned (<45°).  
6. **Velocity penalty** (`-0.01 × speed`) – Discourages excessive speed.  
7. **Oscillation penalty** (`-0.02 × |action_t - action_{t-1}|`) – Penalizes action twitching between timesteps.  
8. **Collision penalty** (`-150`) – Large negative reward when approaching too fast (>0.4 m/s) with AABB overlap detection.  
9. **Boundary penalty** (`-500`) – Nuclear penalty when leaving the arena (out-of-bounds).  
10. **Success bonus** (`+250`) – Strong positive reward when docking is successful (`surface_xy < 3 cm` for ≥5 steps).  
11. **Proximity bonus** (`+4.0`) – Extra reward when very close (3–10 cm) without collision.  
12. **Precision bonus** (`+20.0`) – Awarded for ultra-precise docking (<2 cm).  
13. **Time penalty** (`-0.02`) – Small per-step cost to encourage efficiency.

Total reward is clamped to `[-400, 400]` to prevent extreme values.

This design encourages the agent to dock **quickly but safely**, rather than exploiting collisions or walls. The emphasis on **alignment** is crucial for handling lateral and angular offsets in later curriculum stages.

### 5.4 Policy Network and Visual Encoder

The policy is implemented in pure **PyTorch** (no external RL frameworks at runtime):

* **Encoder:** Configurable CNN defined in `cnn_model.py`.  
* **Actor head:** MLP mapping visual features (256-dim) → `[v_cmd, w_cmd]` (Tanh activation).  
* **Critic head:** MLP mapping visual features → state value estimate.  
* **Action distribution:** Gaussian with learnable global `log_std` parameters (clamped to `[-1.5, 0.2]` for stability).

**SimpleCNN Encoder (default):**

* Lightweight, custom CNN with three Conv2D blocks + BatchNorm + ReLU + MaxPool.  
* Automatically adapts to input size (`480 × 640`).  
* Feature extraction: `[B, 12, 480, 640]` → `[B, 256]`.  
* **Frame stacking support:** Reshapes input to `[B, K, 3, H, W]` where `K = 4`, processes each frame through the same CNN, then mean-pools features over time.  
* Initialised with Kaiming/Xavier schemes for stable gradient flow.  
* **Memory efficient:** Designed to run with 16+ parallel environments on an RTX 3090 without OOM errors.  
* **No ImageNet normalization:** Data comes pre-normalized from environment in `[0, 1]` range.

**MobileNetV3-Small (optional, legacy):**

* Pretrained on ImageNet via `torchvision.models.mobilenet_v3_small`.  
* Provides strong visual feature extraction out of the box.  
* More memory-intensive; useful for **transfer-learning experiments** or encoder comparisons.  
* Currently not actively used due to memory constraints with many parallel environments.

> **Why SimpleCNN as default?**  
> Earlier tests with a pretrained MobileNetV3-Small backbone significantly increased GPU memory usage and occasionally led to out-of-memory errors when training with many parallel environments on the RTX 3090. The SimpleCNN keeps the model compact (~2M parameters), avoids memory issues with 16 environments, and still provides sufficiently rich features for learning the docking behaviour. MobileNetV3-Small remains available in `cnn_model.py` for future experiments targeting transfer learning or detailed encoder comparisons.

### 5.5 PPO Algorithm

The training loop in `scripts/skrl/train_curriculum.py` implements **Proximal Policy Optimization (PPO)** with:

* **GAE(λ)** advantage estimation (`γ = 0.99`, `λ = 0.95`)  
* **Clipped policy objective** (clip ratio = 0.15)  
* **Value clipping** (0.2) for stable critic learning  
* **Stage-dependent entropy regularization** (0.04–0.08) for exploration control  
* **Gradient clipping** (max norm = 0.5)  
* **CPU-based rollout storage** + **GPU mini-batch updates** for memory efficiency with frame stacking  
* **Checkpointing** every 50k steps and at stage transitions  
* **Comprehensive TensorBoard logging** with 36 tracked metrics  

Core hyperparameters are centralized in a `HYPERPARAMS` dictionary, simplifying **future genetic/evolutionary optimisation**.

**Key Implementation Details:**

* **Rollout length:** 64 steps per environment  
* **Mini-batch size:** 64 samples (safe for frame-stacked observations)  
* **PPO epochs:** 4 per update  
* **Learning rate:** `3e-4` (Adam optimizer)

---

## 6. Curriculum Learning

Docking is trained via a **16-stage ultra-gradual curriculum** (`curriculum_manager.py`) that progressively increases task difficulty.

### 6.1 Curriculum Stages

**Stages 0–3: Forward Docking (Distance only)**  
* **Stage 0:** Baby Steps (5–12 cm, perfectly aligned)  
* **Stage 1:** Forward 1 (10–18 cm, perfectly aligned)  
* **Stage 2:** Forward 2 (15–25 cm, perfectly aligned)  
* **Stage 3:** Medium Forward (20–35 cm, perfectly aligned)  

**Stages 4–11: Offset Docking (Distance + Lateral + Angular)**  
* **Stage 4:** Tiny Offset Close (20–30 cm, ±3°, ±3 cm lateral)  
* **Stage 5:** Tiny Offset Medium (20–40 cm, ±6°, ±5 cm lateral)  
* **Stage 6:** Small Offset (20–40 cm, ±9°, ±7 cm lateral)  
* **Stage 7:** Small+ Offset (20–40 cm, ±12°, ±9 cm lateral)  
* **Stage 8:** Medium Offset (20–40 cm, ±15°, ±11 cm lateral)  
* **Stage 9:** Medium+ Offset (20–40 cm, ±18°, ±13 cm lateral)  
* **Stage 10:** Large Offset (20–40 cm, ±21°, ±16 cm lateral)  
* **Stage 11:** Large+ Offset (20–40 cm, ±24°, ±18 cm lateral)  

**Stages 12–13: 180° Turn-Around**  
* **Stage 12:** 180° Close (25–40 cm, facing away, 0 offset)  
* **Stage 13:** 180° Offset (25–40 cm, 0°±10°, ±10 cm lateral)  

**Stages 14–15: Full Autonomy**  
* **Stage 14:** Arena Search (60–120 cm, random yaw)  
* **Stage 15:** Full Autonomy (random position in arena, random yaw)

### 6.2 Stage Progression Logic

A stage is eligible for advancement when:

1. **Minimum steps completed:** ≥15,000 steps in current stage **AND**  
2. **Success rate threshold met:** Stage Success Rate (SSR) reaches threshold **OR**  
3. **Safety valve triggered:** 400,000 steps in current stage (forces advancement to prevent infinite loops)

**Success Rate Thresholds per Stage:**

* Stage 0: 80%  
* Stages 1–4: 70%  
* Stages 5–9: 60%  
* Stages 10–15: 50%  

### 6.3 Stage Mixing

When advancing from stage N to stage N+1:

* **50 short rollouts** (16 steps each) alternate between stages N and N+1.  
* This helps smooth the transition and prevents catastrophic forgetting.  
* The policy receives gradual exposure to the new difficulty level.

### 6.4 Previous-Stage Replay

To prevent catastrophic forgetting:

* **20% of environments** sample from previous stage (stages 0–6)  
* **30% of environments** sample from previous stage (stages 7–9)  
* **40% of environments** sample from previous stage (stages 10–15)  

### 6.5 Stage-Dependent Entropy

Entropy coefficient adapts per stage to balance exploration vs. exploitation:

* **Stages 0–1:** 0.08 (high exploration for learning basics)  
* **Stages 2–3:** 0.07 (still high exploration)  
* **Stages 4–7:** 0.06 (maintained exploration for offset challenges)  
* **Stages 8+:** 0.04 (reduced for policy refinement)  

This dynamic schedule ensures sufficient exploration during challenging offset stages while allowing exploitation during simpler or refinement phases.

---

## 7. Training Workflow

### 7.1 Launching Training (Headless)

From the repository root:

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/skrl/train_curriculum.py \
  --num_envs 16 \
  --steps 60000000 \
  --lr 3e-4 \
  --headless
