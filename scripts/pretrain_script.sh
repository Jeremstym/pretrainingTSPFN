#!/bin/bash
#SBATCH --partition=heavy
#SBATCH --nodelist=dc
#SBATCH --job-name=TSPFN-pretraining
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/TRASH_TEST' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_BIGpretraining_v3' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2.5' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2-SinglePE' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2-LearnedPE' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=learned
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2-LearnedPE' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=learned ckpt="/data/stympopper/TSPFN_v2-LearnedPE/checkpoints/epoch\=499-step\=37500_FL.ckpt" trainer.max_epochs=1000

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/Baseline' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/Baseline-zscoring' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-Real' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-csv task.time_series_positional_encoding=rope+channel
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2-RoPE' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=rope task.optim.scheduler=null ckpt="/data/stympopper/TSPFN_v2-RoPE/checkpoints/epoch\=499-step\=37500_FL.ckpt" trainer.max_epochs=1000

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTraining' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ShortBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 trainer.max_epochs=100
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/SinusBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=sinusoidal trainer.max_epochs=100
