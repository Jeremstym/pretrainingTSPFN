#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --nodelist=punk
#SBATCH --job-name=CauKer-generating
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

poetry run python /home/stympopper/CauKer/CauKer.py -N 2000000 -L 512 -F 4 -P 6 -M 18 -O /data/stympopper/CauKer2M/CauKer2M.arrow
