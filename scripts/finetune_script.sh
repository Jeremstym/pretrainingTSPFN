#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --cpus-per-task=5
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
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune-collatefn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt

#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50 task.time_series_num_channels=23
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt trainer.max_epochs=50

#//#! all collate_fn below
#// Remove collate fn for TabPFNv2.5 finetuning
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
#// poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-newconvol/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-warmup/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-warmup/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-hardsubsampling/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=50

#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/filteredTUEV-Finetune-TabPFNv2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/filteredTUEV-Finetune-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100


poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/filteredTUEV-Finetune-TabPFNv2-labram/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 trainer.max_epochs=100
#! TESTING with ckpts

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune/seed42/checkpoints/epoch\=3-step\=3052.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-TSPFNFinetune/seed42/checkpoints/epoch\=11-step\=9156.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/FullTUEV-Finetune/seed42/checkpoints/epoch\=1-step\=988.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/FullTUEV-TSPFNFinetune/seed42/checkpoints/epoch\=1-step\=988.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed42/checkpoints/epoch\=3-step\=1524.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-longwindow/seed42/checkpoints/epoch\=10-step\=4191.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-collatefn-TabPFNv2.5/seed42/checkpoints/epoch\=6-step\=2667.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-newconvol/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-newconvol/seed42/checkpoints/epoch\=3-step\=1524.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5/seed42/checkpoints/epoch\=4-step\=3810.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2/seed42/checkpoints/epoch\=3-step\=3048.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-hardsubsampling/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-hardsubsampling/seed42/checkpoints/epoch\=0-step\=762.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-warmup/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-warmup/seed42/checkpoints/epoch\=13-step\=10668.ckpt' train=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-filteredTUEV/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2-filteredTUEV/seed42/checkpoints/epoch\=86-step\=9396.ckpt' train=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-filteredTUEV/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 'ckpt=/data/stympopper/TSPFN_results/TUEV-Finetune-TabPFNv2.5-filteredTUEV/seed42/checkpoints/epoch\=14-step\=1620.ckpt' train=False
