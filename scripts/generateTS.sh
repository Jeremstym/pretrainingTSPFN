#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=CauKer-generating
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

module load cuda/11.1 2>/dev/null || module load cuda/11.1.1 2>/dev/null

export CUDA_PATH=/usr/local/cuda-11.1
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

cd /home/stympopper/pretrainingTSPFN

poetry run python /home/stympopper/CauKer/CauKer.py -N 2000000 -L 512 -F 4 -P 6 -M 18 -O /data/stympopper/CauKer2M/CauKer2M.arrow
