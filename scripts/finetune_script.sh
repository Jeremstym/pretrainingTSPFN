#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=8G
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
#CHECK poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42
#CHECK poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune-collatefn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt trainer.max_epochs=100

#//#! all collate_fn below
#! Remove collate fn for TabPFNv2.5 finetuning
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50

#! TESTING with ckpts

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune/seed42/checkpoints/epoch\=3-step\=3052.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed42/checkpoints/epoch\=11-step\=9156.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed42/checkpoints/epoch\=1-step\=988.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed42/checkpoints/epoch\=1-step\=988.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed42/checkpoints/epoch\=3-step\=1524.ckpt' train=False

