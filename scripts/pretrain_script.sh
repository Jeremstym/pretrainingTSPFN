#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=TUEV-processing
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TRASH_TEST' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_BIGpretraining_v3' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTraining' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ShortBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 trainer.max_epochs=100
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/SinusBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=sinusoidal trainer.max_epochs=100

#! PRORCESSING TUEV

python -m ~/pretrainingTSPFN/data/tuev_preprocessing.py