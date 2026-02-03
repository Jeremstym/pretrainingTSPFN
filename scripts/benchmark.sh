#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --nodelist=kavinsky
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tabpfn-v2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-ecg5000 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2-dummy/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-dummy.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True ckpt="/data/stympopper/TSPFN_v2.5/checkpoints/epoch\=221-step\=16650.ckpt" task.embed_dim=512
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2-LearnedPE-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-LearnedPE.pt" task.time_series_positional_encoding=learned
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2-RoPE/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt" task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tspfnv2-RoPE-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt" task.time_series_positional_encoding=rope

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/seed${seed}' +experiment=baselines/baseline seed=42 train=True test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TCN/seed${seed}' +experiment=baselines/baseline data=finetuning-esr seed=42 train=True test=True


# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True task.embed_dim=512
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tabpfnv2-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2.5/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-5.pt" task.embed_dim=512
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2-RoPE/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt" task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2-RoPE-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt" task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2-LearnedPE/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-LearnedPE.pt" task.time_series_positional_encoding=learned
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-tspfnv2-LearnedPE-scaler/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-LearnedPE.pt" task.time_series_positional_encoding=learned

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/abide-tabpfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/abide-tspfn/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/abide-tspfnv2-RoPE/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt" 
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/abide-tspfnv2-LearnedPE/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-LearnedPE.pt" task.time_series_positional_encoding=learned
