#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=CubePFN3-FineTune
####BATCH --cpus-per-task=5
####SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecg200-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecg200 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecg5000-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecg5000 seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope
# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSPFN-Benchmark/ecgfivedays-TSPFN/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecgfivedays seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" task.time_series_positional_encoding=cwpe+rope

UCR2019_univariate=(
    # "ACSF1"
    # "Adiac"
    # "AllGestureWiimoteX"
    # "AllGestureWiimoteY"
    # "AllGestureWiimoteZ"
    # "ArrowHead"
    # "Beef"
    # "BeetleFly"
    # "BirdChicken"
    # "BME"
    # "Car"
    # "CBF"
    # "Chinatown"
    # "ChlorineConcentration"
    # "CinCECGTorso"
    # "Coffee"
    # "Computers"
    # "CricketX"
    # "CricketY"
    # "CricketZ"
    # "Crop"
    # "DiatomSizeReduction"
    # "DistalPhalanxOutlineAgeGroup"
    # "DistalPhalanxOutlineCorrect"
    # "DistalPhalanxTW"
    # "DodgerLoopDay"
    # "DodgerLoopGame"
    # "DodgerLoopWeekend"
    # "Earthquakes"
    # "ECG200"
    # "ECG5000"
    # "ECGFiveDays"
    # "ElectricDevices"
    # "EOGHorizontalSignal"
    # "EOGVerticalSignal"
    # "EthanolLevel"
    "FaceAll"
    "FaceFour"
    "FacesUCR"
    "FiftyWords"
    "Fish"
    "FordA"
    "FordB"
    "FreezerRegularTrain"
    "FreezerSmallTrain"
    "Fungi"
    "GestureMidAirD1"
    "GestureMidAirD2"
    "GestureMidAirD3"
    "GesturePebbleZ1"
    "GesturePebbleZ2"
    "GunPoint"
    "GunPointAgeSpan"
    "GunPointMaleVersusFemale"
    "GunPointOldVersusYoung"
    "Ham"
    "HandOutlines"
    "Haptics"
    "Herring"
    "HouseTwenty"
    "InlineSkate"
    "InsectEPGRegularTrain"
    "InsectEPGSmallTrain"
    "InsectWingbeatSound"
    "ItalyPowerDemand"
    "LargeKitchenAppliances"
    "Lightning2"
    "Lightning7"
    "Mallat"
    "Meat"
    "MedicalImages"
    "MelbournePedestrian"
    "MiddlePhalanxOutlineCorrect"
    "MiddlePhalanxOutlineAgeGroup"
    "MiddlePhalanxTW"
    "MixedShapesRegularTrain"
    "MixedShapesSmallTrain"
    "MoteStrain"
    "NonInvasiveFetalECGThorax1"
    "NonInvasiveFetalECGThorax2"
    "OliveOil"
    "OSULeaf"
    # "PhalangesOutlinesCorrect"
    # "Phoneme"
    # "PickupGestureWiimoteZ"
    # "PigAirwayPressure"
    # "PigArtPressure"
    # "PigCVP"
    # "PLAID"
    # "Plane"
    # "PowerCons"
    # "ProximalPhalanxOutlineCorrect"
    # "ProximalPhalanxOutlineAgeGroup"
    # "ProximalPhalanxTW"
    # "RefrigerationDevices"
    # "Rock"
    # "ScreenType"
    # "SemgHandGenderCh2"
    # "SemgHandMovementCh2"
    # "SemgHandSubjectCh2"
    # "ShakeGestureWiimoteZ"
    # "ShapeletSim"
    # "ShapesAll"
    # "SmallKitchenAppliances"
    # "SmoothSubspace"
    # "SonyAIBORobotSurface1"
    # "SonyAIBORobotSurface2"
    # "StarLightCurves"
    # "Strawberry"
    # "SwedishLeaf"
    # "Symbols"
    # "SyntheticControl"
    # "ToeSegmentation1"
    # "ToeSegmentation2"
    # "Trace"
    # "TwoLeadECG"
    # "TwoPatterns"
    # "UMD"
    # "UWaveGestureLibraryAll"
    # "UWaveGestureLibraryX"
    # "UWaveGestureLibraryY"
    # "UWaveGestureLibraryZ"
    # "Wafer"
    # "Wine"
    # "WordSynonyms"
    # "Worms"
    # "WormsTwoClass"
    # "Yoga"
)

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-MedPretrained-FineTune/seed\${seed}" \
#         +experiment=finetuningTSPFN/cubepfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=True \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubePFN3-pretrained-attchanCLSAVG.ckpt" \
#         strict=False
# done

for dataset in "${UCR2019_univariate[@]}"; do
    poetry run tspfn-pretrain \
        "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-FineTune/seed\${seed}" \
        +experiment=finetuningTSPFN/cubepfn3-finetuning \
        data=benchmark/evaluating-ucrunivariate \
        data.dataset="$dataset" \
        task.adaptable_metrics=True \
        seed=42 \
        +dataset="$dataset" \
        train=True \
        test=True
done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-MantisV2-FineTune/seed\${seed}" \
#         +experiment=baselines/mantis_v2 \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         data.mantis_training=True \
#         seed=42 \
#         +dataset="$dataset" \
#         task.finetuning=True \
#         train=False \
#         test=True
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-Mantis-FineTune/seed\${seed}" \
#         +experiment=finetuningTSPFN/cubepfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=True \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubepfn-pretrained-mantis-v2.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-Mantis-NoDiff/seed\${seed}" \
#         +experiment=finetuningTSPFN/cubepfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubepfn-pretrained-mantis-nodiff.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-Mantis/seed\${seed}" \
#         +experiment=finetuningTSPFN/cubepfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubepfn-pretrained-mantis-v2.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN-mantis-CauKer2M/seed\${seed}" \
#         +experiment=finetuningTSPFN/cubepfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/CubePFN-mantis-pretrained.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-CauKer100K/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubePFN-pretrained-cauker100K.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-CauKer100K-noFFT/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubePFN-pretrained-cauker100K-noFFT.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-contrastive-last/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/CubePFN-contrastive-last.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubePFN3-contrastive/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/CubePFN-contrastive-best.ckpt" \
#         strict=False
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-CubeICL-v3-attchanCLSAVG/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         ckpt="/home/stympopper/pretrainingTSPFN/ckpts/cubeICL3-pretrained-attchanCLSAVG.ckpt"
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-TSPFN-v3/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn3-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True
# done

# for dataset in "${UCR2019_univariate[@]}"; do
#     poetry run tspfn-pretrain \
#         "hydra.run.dir=/data/stympopper/TSPFN-Benchmark/UCRUnivariate/${dataset}-TSPFN/seed\${seed}" \
#         +experiment=finetuningTSPFN/tspfn-finetuning \
#         data=benchmark/evaluating-ucrunivariate \
#         data.dataset="$dataset" \
#         task.adaptable_metrics=True \
#         seed=42 \
#         +dataset="$dataset" \
#         train=False \
#         test=True \
#         updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE+CWPE-zscoring-5CHANS+hirid-nowarmup-shuffle-2.pt" \
#         task.time_series_positional_encoding=cwpe+rope
# done

# TSICL test

# poetry run tspfn-pretrain 'hydra.run.dir=/data/stympopper/TSICL-Benchmark/ecgfivedays-TSICL/seed${seed}' +experiment=finetuningTSPFN/tspfn-finetuning data=benchmark/evaluating-ecgfivedays seed=42 train=False test=True updated_pfn_path="/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-ICL.pt" task.time_series_positional_encoding=cwpe+rope
