#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --nodes=1
#SBATCH --nodelist=kavinsky
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost' +experiment=baselines/xgboost seed=42 train=False test=True
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-xgboost' +experiment=baselines/xgboost seed=42 train=False test=True