#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --nodes=1
#SBATCH --nodelist=kavinsky
#SBATCH --gpus-per-node=1
#SBATCH --job-name=Mantis-tokenizer
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-mantis-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-mantis-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-mantis-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-eos seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-mantis-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-eicucrd seed=42 train=False test=True
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-mantis-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-cpsc seed=42 train=False test=True