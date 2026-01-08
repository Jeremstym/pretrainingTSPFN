#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-zeroshot/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt train=False
