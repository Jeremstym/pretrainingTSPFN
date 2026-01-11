#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=TUEV-finetuning
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-zeroshot/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt

poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt trainer.max_epochs=100

#! TESTING with ckpts

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune/seed42/checkpoints/epoch\=3-step\=3052.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed42/checkpoints/epoch\=11-step\=9156.ckpt' train=False