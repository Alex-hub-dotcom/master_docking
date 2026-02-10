# TEKO files. How to use them.

## Objective

Final experiments for the thesis, comparing:
1. **Vision Optimal** - Policy with optimal hyperparameters
2. **Vision Optuna** - Seach for hyperparameter combinations for grafical comparison
3. **State Full** - Baseline oracle with full information

## Structure

```
Final_trial/
├── train_vision_optimal.py    # Vision with static hyperparameter
├── train_vision_optuna.py     # Optuna search 
├── train_state_full.py        # State with full information
├── run_vision_optimal.sh      # SLURM script
├── run_vision_optuna.sh       # SLURM script 
├── run_state_full.sh          # SLURM script
├── launch_all.sh              # Launch all experiments
└── README.md                  # This file
```

## Hyperparameters

| Parameter     | Value   |
|---------------|---------|
| learning_rate | 0.000162|
| entropy_coef  | 0.00622 |
| gae_lambda    | 0.9396  |
| batch_size    | 1024    |
| epochs        | 5       |
| aux_yaw_coef  | 0.308   |

## Why can the State model reach S41?

The original State setup only provided privileged information (dx, dy, dz, yaw_err) to the CRITIC.
The actor only received IMU data (velocities).

**Problema:** Without knowing where the target is, the actor cannot dock from approach angles >90°.

**Solução (State Full):** Provide full information to the ACTOR:
- State: 10D [dx, dy, dz, yaw_err, vx, vy, vz, wx, wy, wz]
- The actor "knows" where the target is even without seeing it

This serves as a **baseline oracle** to prove the task is solvable.
If Vision Optimal matches this result, that becomes the thesis contribution!

## Expected Comparison

| Model         | Max Stage  | Actor Information |  Time  |
|---------------|------------|-------------------|--------|
| State Full    | S41 (180°) | Complete (10D)    | ~6-12h |
| Vision Optimal| S41 (180°) | Camera + IMU      | ~7-10h |
| Vision Optuna | S27 (75°)  | Camera + IMU      | ~10h   |

## Logging CONFIRM TJHIS PART !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### TensorBoard
```bash
tensorboard --logdir=/home/schux00/tensorboard --port=6006
```

### CSV
Ficheiros em `/home/schux00/logs/*.csv`

## Como lançar

```bash
# Copiar scripts para o cluster
cp -r /home/claude/Final_trial /home/schux00/teko/scripts/

# Tornar executáveis
chmod +x /home/schux00/teko/scripts/Final_trial/*.sh

# Lançar todos
/home/schux00/teko/scripts/Final_trial/launch_all.sh

# Ou individualmente
sbatch /home/schux00/teko/scripts/Final_trial/run_vision_optimal.sh
sbatch /home/schux00/teko/scripts/Final_trial/run_vision_optuna.sh
sbatch /home/schux00/teko/scripts/Final_trial/run_state_full.sh
```

## Monitorização

```bash
# Ver jobs
squeue -u schux00

# Ver logs em tempo real
tail -f /home/schux00/logs/vision_optimal_*.out
tail -f /home/schux00/logs/state_full_*.out

# Ver Optuna trials
tail -f /home/schux00/logs/vision_optuna_*.out
```

## Resultados para a Tese

Os gráficos gerados mostram:

1. **Curriculum Progression** - Stage vs Steps (Vision vs State)
2. **Success Rate** - SSR ao longo do treino
3. **Hyperparameter Search** - Pareto front do Optuna
4. **Comparação Vision vs State** - Prova que visão = state com info completa

Autor: Alexandre Schleier Neves da Silva
Data: Janeiro 2026
