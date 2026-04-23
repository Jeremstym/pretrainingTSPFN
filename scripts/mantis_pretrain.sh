#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --nodelist=hard
#SBATCH --job-name=Mantis-pretraining
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/Mantis/pretrain-25seq-192dim' +experiment=pretrainingMantis/mantis-pretraining seed=42 data=pretraining-data-mantis
