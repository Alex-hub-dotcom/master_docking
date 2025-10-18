# TEKO Docking System

This repository is part of the master's thesis project *"Adaptive Cooperation for Agricultural Robotic Swarms: Docking of Modular Robotic Units in Precision Agriculture Using Reinforcement Learning"*, developed at the University of Hohenheim.

The project investigates how small-scale agricultural robots can physically connect to form modular units capable of performing tasks that would be difficult or impossible for a single robot to accomplish. The research focuses on the development of a reinforcement learning-based docking system using simulated environments and real-world hardware, with the TEKO robot as the experimental testbed.

## Objectives

- Design and implement a reliable **docking mechanism** between modular robots.
- Develop a **reinforcement learning agent** capable of learning physical coupling behavior.
- Explore **evolutionary algorithms** to improve robustness and exploration during training.
- Validate the approach in **simulation** using NVIDIA Isaac Lab.
- Address the **sim-to-real transfer** challenges for deployment in real-world agricultural scenarios.

## Repository Contents

- `/documents`: Robot models (USD, URDF, STL files)
- `/scripts`: Code for agent training, camera streaming, and environment setup
- `/logs`: Training logs and checkpoint files for PPO-based reinforcement learning
- `README.md`: This documentation file

## Technologies Used

- **NVIDIA Isaac Lab** for simulation and physics modeling
- **Python** and **PyTorch** for reinforcement learning (PPO, hybrid EA-RL)
- **skrl** library for RL agents
- **USD and URDF** for robot modeling
- **ROS 2 (planned)** for hardware-level implementation and control

## Contact

For questions, feedback, or collaboration opportunities, please contact:

**Alexandre Schleier Neves da Silva**  
M.Sc. Environmental Protection and Agricultural Food Production  
University of Hohenheim  
alexandre.schleiernevesdasilva@uni-hohenheim.de
