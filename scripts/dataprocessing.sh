#!/bin/bash
#SBATCH --partition=electronic,hard,funky
#SBATCH --job-name=TUEV-processing
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

#! PRORCESSING TUEV

# poetry run python ~/pretrainingTSPFN/data/tuev_preprocessing.py
# poetry run python ~/pretrainingTSPFN/data/tuab_preprocessing.py
# poetry run python ~/pretrainingTSPFN/data/tuev_cleaning.py
poetry run python /data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/eicu_makelabels.py