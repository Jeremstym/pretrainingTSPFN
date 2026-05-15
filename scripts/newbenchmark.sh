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

poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecg200-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecg200 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecg5000-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecgfivedays-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecgfivedays seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope
