#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=TSPFN-Baselines
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
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-xgboost' data=evaluating-eicucrd +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/blink-xgboost' data=evaluating-blink +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-xgboost' data=evaluating-eos +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/atrialfibri-xgboost' data=evaluating-atrialfibri +experiment=baselines/xgboost seed=42 train=False test=True
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-xgboost' data=evaluating-cpsc +experiment=baselines/xgboost seed=42 train=False test=True

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/seed${seed}' +experiment=baselines/baseline seed=42 data=finetuning-ecg5000 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TCN/seed${seed}' +experiment=baselines/baseline seed=42 data=finetuning-esr train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eos seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/atrialfibri-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-atrialfibri seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TCN-full/seed${seed}' +experiment=baselines/baseline data=finetuning-cpsc seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-esr seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eos seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-patchTST-full/seed${seed}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-minirocket-full/seed${seed}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-minirocket-full/seed${seed}' +experiment=baselines/baseline data=finetuning-esr task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-minirocket-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eos task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-minirocket-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-minirocket-full/seed${seed}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-labram-full/seed${seed}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-labram-full/seed${seed}' +experiment=baselines/baseline data=finetuning-esr task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-labram-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eos task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-labram-full/seed${seed}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-labram-full/seed${seed}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0

# Supsize spanning

# for supsize in 50, 100, 250, 500, 750, 1000; do
#   for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/fold${fold}-sups${supsize}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} data.fold=${fold} +fold=${fold}
#   done
# done

for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tabpfn/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-ecg5000 train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost/multisups${supsize}' +experiment=baselines/xgboost data=evaluating-ecg5000 seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/multisups${supsize}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-labram/multisups${supsize}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
  done

for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-eicucrd seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-tabpfn/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-eicucrd train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-xgboost/multisups${supsize}' +experiment=baselines/xgboost data=evaluating-eicucrd seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN/multisups${supsize}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-labram/multisups${supsize}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
  done

for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-cpsc seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-tabpfn/multisup${supsize}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-cpsc train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-xgboost/multisups${supsize}' +experiment=baselines/xgboost data=evaluating-cpsc seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TCN/multisups${supsize}' +experiment=baselines/baseline data=finetuning-cpsc seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-labram/multisups${supsize}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize}
  done

for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-minirocket/multisup${supsize}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-minirocket/multisup${supsize}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-minirocket/multisup${supsize}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize}
  done

for fold in 0 1 2 3 4; do
  for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-minirocket/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-minirocket/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-minirocket/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
  done
done

# ----------------------------

for fold in 0 1 2 3 4; do
  for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-tabpfn/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-ecg5000 train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost/multisup${supsize}/fold${fold}' +experiment=baselines/xgboost data=evaluating-ecg5000 seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-labram/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
  done
done

for fold in 0 1 2 3 4; do
  for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-eicucrd seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-tabpfn/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-eicucrd train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-xgboost/multisup${supsize}/fold${fold}' +experiment=baselines/xgboost data=evaluating-eicucrd seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-labram/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
  done
done

for fold in 0 1 2 3 4; do
  for supsize in 50 100 250 500 750 1000; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TSPFNFM-RoPE+CWPE-zscoring-5chans-v2/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning data=evaluating-cpsc seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-tabpfn/multisup${supsize}/fold${fold}' +experiment=finetuningTSPFN/tspfn-finetuning seed=42 data=evaluating-cpsc train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-xgboost/multisup${supsize}/fold${fold}' +experiment=baselines/xgboost data=evaluating-cpsc seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TCN/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-labram/multisup${supsize}/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
  done
done

for fold in 0 1 2 3 4; do
  for supsize in 50 100; do
        poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-xgboost/multisup${supsize}/fold${fold}' +experiment=baselines/xgboost data=evaluating-ecg5000 seed=42 train=False test=True data.support_size=${supsize} +supsize=${supsize} data.fold=${fold} +fold=${fold}
  done
done



# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-esr seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-eos seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-patchTST/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-patchTST/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-patchTST/fold${fold}' +experiment=baselines/baseline data=finetuning-esr task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-patchTST/fold${fold}' +experiment=baselines/baseline data=finetuning-eos task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-minirocket/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket data.support_size=500 data.fold=${fold} +fold=${fold} +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-minirocket/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket data.support_size=500 data.fold=${fold} +fold=${fold} +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-minirocket/fold${fold}' +experiment=baselines/baseline data=finetuning-esr task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket data.support_size=500 data.fold=${fold} +fold=${fold} +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
#   done

# for fold in 0 1 2; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-minirocket/fold${fold}' +experiment=baselines/baseline data=finetuning-eos task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket data.support_size=500 data.fold=${fold} +fold=${fold} +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
#   done

# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-minirocket/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=minirocket seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 task.baseline_name=minirocket data.support_size=500 data.fold=${fold} +fold=${fold} +trainer.num_sanity_val_steps=0 trainer.enable_model_summary=False
#   done


# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-TCN/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done
# for fold in 0 1 2 3 4; do
#     poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-patchTST/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=patchTST seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
#   done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/cpsc-labram/fold${fold}' +experiment=baselines/baseline data=finetuning-cpsc task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eicu-labram/fold${fold}' +experiment=baselines/baseline data=finetuning-eicucrd task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/eos-labram/fold${fold}' +experiment=baselines/baseline data=finetuning-eos task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/esr-labram/fold${fold}' +experiment=baselines/baseline data=finetuning-esr task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done

for fold in 0 1 2 3 4; do
    poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN_results/ecg5000-labram/fold${fold}' +experiment=baselines/baseline data=finetuning-ecg5000 task/model/encoder=labram task/model/prediction_head=prediction task.embed_dim=200 seed=42 train=True test=True use_last=True trainer.max_epochs=15 +trainer.limit_val_batches=0.0 data.support_size=500 data.fold=${fold} +fold=${fold}
  done