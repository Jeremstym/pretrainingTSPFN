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

for supsize in 10 50 100 500 1000; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TSPFNFM-RoPE-zscoring-5chans/supsize${supsize}/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring-5CHANS-nowarmup.pt" task.time_series_positional_encoding=rope data.support_size=${supsize} +supsize=${supsize}
    # poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TSPFNFM-Baseline-zscoring-5chans/supsize${supsize}/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-esr seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFNFM_Baseline-zscoring-5CHANS.pt" task.time_series_positional_encoding=rope data.support_size=${supsize} +supsize=${supsize}
    # poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-xgboost/supsize${supsize}' data=evaluating-esr +experiment=baselines/xgboost seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize}
  done

for supsize in 10 50 100 500 1000; do
    # poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TSPFNFM-RoPE-zscoring-5chans/supsize${supsize}/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring-5CHANS-nowarmup.pt" task.time_series_positional_encoding=rope data.support_size=${supsize} +supsize=${supsize}
    # poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TSPFNFM-Baseline-zscoring-5chans/supsize${supsize}/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFNFM_Baseline-zscoring-5CHANS.pt" task.time_series_positional_encoding=rope data.support_size=${supsize} +supsize=${supsize}
    # poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost/supsize${supsize}' data=evaluating-ecg5000 +experiment=baselines/xgboost seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize}
  done
