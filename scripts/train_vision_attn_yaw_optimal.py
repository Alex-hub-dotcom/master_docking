#!/usr/bin/env python3
"""
TEKO Vision + Attention + YawAux - OPTIMAL CONFIG
Hyperparameters from Optuna Trial 80 (reached S41/180°)
"""
# Copiar do debug mas com estes valores:
CONFIG = {
    "max_steps": 200_000_000,
    "max_hours": 168,  # 7 dias
    
    # OPTIMAL from Optuna Trial 80
    "learning_rate": 0.000162,
    "entropy_coef": 0.00622,
    "gae_lambda": 0.9396,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 5,
    "batch_size": 1024,
    
    "aux_yaw_coef": 0.308,
    
    "num_envs": 120,
    "rollout_len": 128,
    
    "advance_threshold": 0.75,
    "min_steps_before_advance": 200_000,
    "max_stage": 41,
    
    "log_interval": 50_000,
    "save_interval": 2_000_000,
}
