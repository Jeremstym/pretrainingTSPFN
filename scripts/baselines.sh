#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TUEV-finetuning
#SBATCH --nodes=1
#SBATCH --nodelist=kavinsky
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost' data=evaluating-ecg5000 +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-xgboost' data=evaluating-esr +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/abide-xgboost' +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/orchid-xgboost' data=evaluating-orchid +experiment=baselines/xgboost seed=42 train=False test=True
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-xgboost' data=evaluating-eicucrd +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/blink-xgboost' data=evaluating-blink +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-xgboost' data=evaluating-eos +experiment=baselines/xgboost seed=42 train=False test=True

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/seed${seed}' +experiment=baselines/baseline seed=42 data=finetuning-ecg5000 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TCN/seed${seed}' +experiment=baselines/baseline seed=42 data=finetuning-esr train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eos seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0

poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-esr seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-eos seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done