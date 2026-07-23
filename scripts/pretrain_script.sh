#!/bin/bash
#SBATCH --partition=heavy
###SBATCH --nodelist=ac
#SBATCH --job-name=ConvolPFN3-pretraining
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
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/Baseline-zscoring-5CHANS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/Baseline-zscoring-5CHANS+hird-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=none
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-Real' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-csv task.time_series_positional_encoding=rope+channel
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS-warmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS-nowarmup2' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring-2CHANS-nowarmup.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-2CHANS-new-warmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-3CHANS-warmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-5CHANS-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-5CHANS-nowarmup-2' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring-5CHANS-nowarmup-v2.pt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE-zscoring-5CHANS+hird-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-zscoring-5CHANS-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-zscoring-5CHANS-nowarmup-shuffle' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope
# ! poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope
#? poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSICL_FM/RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope task.model.encoder.random_init=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/RoPE+CWPE-noPFNPE-zscoring-5CHANS-nowarmup' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_v2-RoPE' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=rope task.optim.scheduler=null ckpt="/data/stympopper/TSPFN_v2-RoPE/checkpoints/epoch\=499-step\=37500_FL.ckpt" trainer.max_epochs=1000

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTraining' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/BatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ShortBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 trainer.max_epochs=100
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/SinusBatchTrainingNOHUGEDS' +experiment=pretrainingTSPFN/tspfn-pretraining seed=42 task.time_series_positional_encoding=sinusoidal trainer.max_epochs=100

#! Custom tokenizer (eg. Mantis) pretraining runs

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_FM/TSPFN-Mantis' +experiment=pretrainingTSPFN/tspfn-pretraining task.use_tokenizer=True seed=42 data=pretraining-data task.time_series_positional_encoding=cwpe+rope

#! Pretraining CubePFN for TabPFN v3

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TabPFN-v3-AttentionChannel-CLS' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TabPFN3-MedPretrained' +experiment=pretrainingTSPFN/tspfn3-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TabPFN3-Reg-MedPretrained' +experiment=pretrainingTSPFN/tspfn3-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TabICL-v3-AttentionChannelCLS-AvgCLS' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42 task.model.encoder.use_checkpoint=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TabICL-v3-AttentionChannelCLS-AvgCLS-part2' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42 task.model.encoder.use_checkpoint=False ckpt="/data/stympopper/CubePFN_FM/TabICL-v3-AttentionChannelCLS-AvgCLS/checkpoints/epoch\=100-step\=1515.ckpt"
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN3-MedPretrained' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/TRASH-TEST' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN3-Reg-MedPretrained' +experiment=pretrainingTSPFN/cubepfn3-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-SSL-Cauker2M' +experiment=pretrainingTSPFN/cubepfn3-contrastive-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-SSL-Cauker100K' +experiment=pretrainingTSPFN/cubepfn3-contrastive-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-SSL-Cauker100K-noFFT' +experiment=pretrainingTSPFN/cubepfn3-contrastive-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-MantisPretraining-FFT' +experiment=pretrainingTSPFN/cubepfn3-mantis-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-MantisPretraining-v2' +experiment=pretrainingTSPFN/cubepfn3-mantis-pretraining seed=42
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN-MantisPretraining-noDiff' +experiment=pretrainingTSPFN/cubepfn3-mantis-pretraining seed=42

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/CubePFN2p5-MedPretrained' +experiment=pretrainingTSPFN/cubepfn2p5-pretraining seed=42 task.model.encoder.use_checkpoint=False
poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/CubePFN_FM/ConvolPFN3-MedPretrained' +experiment=pretrainingTSPFN/convolpfn3-pretraining seed=42
