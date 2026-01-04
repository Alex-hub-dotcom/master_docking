#!/bin/bash
#SBATCH --job-name=create_studies
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/schux00/logs/create_studies_%j.out
#SBATCH --error=/home/schux00/logs/create_studies_%j.err

mkdir -p /home/schux00/optuna

cd /home/schux00/teko

# Create vision study
echo "Creating vision study..."
enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -c "
import sys
sys.path.insert(0, '/workspace/teko/scripts/optuna')
from train_optuna_vision import make_storage, create_study, OPTUNA_CONFIG
storage = make_storage(OPTUNA_CONFIG['storage_path'])
create_study(OPTUNA_CONFIG['study_name'], storage)
print('[OK] Vision study created')
"

# Create state study
echo "Creating state study..."
enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -c "
import sys
sys.path.insert(0, '/workspace/teko/scripts/optuna')
from train_optuna_state import make_storage, create_study, OPTUNA_CONFIG
storage = make_storage(OPTUNA_CONFIG['storage_path'])
create_study(OPTUNA_CONFIG['study_name'], storage)
print('[OK] State study created')
"

echo "Done!"
ls -la /home/schux00/optuna/
