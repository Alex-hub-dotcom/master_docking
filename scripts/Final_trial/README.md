# TEKO Final Experiments

## Objetivo

Experimentos finais para a tese comparando:
1. **Vision Optimal** - Policy com hyperparâmetros óptimos do Trial 80
2. **Vision Optuna** - Busca de hyperparâmetros para gráficos comparativos
3. **State Full** - Baseline oracle com informação completa

## Estrutura

```
Final_trial/
├── train_vision_optimal.py    # Vision com hyperparâmetros Trial 80
├── train_vision_optuna.py     # Optuna search (novo study v2)
├── train_state_full.py        # State com info completa ao actor
├── run_vision_optimal.sh      # SLURM script
├── run_vision_optuna.sh       # SLURM script (array 0-2)
├── run_state_full.sh          # SLURM script
├── launch_all.sh              # Lança todos os experimentos
└── README.md                  # Este ficheiro
```

## Hyperparâmetros Óptimos (Trial 80)

| Parâmetro | Valor |
|-----------|-------|
| learning_rate | 0.000162 |
| entropy_coef | 0.00622 |
| gae_lambda | 0.9396 |
| batch_size | 1024 |
| epochs | 5 |
| aux_yaw_coef | 0.308 |

## Por que o State pode chegar a S41?

O State original só dava informação privilegiada (dx, dy, dz, yaw_err) ao **CRITIC**.
O actor só recebia IMU (velocidades).

**Problema:** Sem saber onde está o alvo, o actor não consegue fazer docking >90°.

**Solução (State Full):** Dar informação completa ao **ACTOR**:
- Estado: 10D [dx, dy, dz, yaw_err, vx, vy, vz, wx, wy, wz]
- O actor "sabe" onde está o alvo mesmo sem ver

Isto serve como **baseline oracle** para provar que a tarefa É solucionável.
O Vision Optimal igualar este resultado = contribuição da tese!

## Comparação Esperada

| Modelo | Max Stage | Info ao Actor | Tempo |
|--------|-----------|---------------|-------|
| State Full | S41 (180°) | Completa (10D) | ~6-12h |
| Vision Optimal | S41 (180°) | Câmera + IMU | ~7-10h |
| State IMU (antigo) | S27 (75°) | Só IMU (6D) | ~10h |

## Logging

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
