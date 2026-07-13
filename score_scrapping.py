import os
import sys
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

label_10_filter = [
    "Crop",
    "PLAID",
    "PigArtPressure",
    "ShapesAll",
    "GestureMidAirD3",
    "NonInvasiveFetalECGThorax1",
    "InsectWingbeatSound",
    "PigCVP",
    "GestureMidAirD1",
    "CricketX",
    "CricketZ",
    "GestureMidAirD2",
    "FacesUCR",
    "Fungi",
    "Adiac",
    "NonInvasiveFetalECGThorax2",
    "Phoneme",
    "FiftyWords",
    "CricketY",
    "EOGHorizontalSignal",
    "PigAirwayPressure",
    "SwedishLeaf",
    "FaceAll",
    "EOGVerticalSignal",
    "WordSynonyms",
]


def simplify_csv_pandas(df: pd.DataFrame) -> pd.DataFrame:
    name_mapping = {
        # --- Multiclass Metrics ---
        "test/MulticlassAccuracy/classification": "accuracy",
        "test/MulticlassAUROC/classification": "auroc",
        "test/MulticlassAveragePrecision/classification": "average_precision",
        "test/MulticlassF1Score/classification": "f1_score",
        "test/MulticlassCohenKappa/classification": "cohen_kappa",
        "test/MulticlassRecall/classification": "recall",
        # --- Binary Metrics ---
        "test/BinaryAccuracy/classification": "accuracy",
        "test/BinaryAUROC/classification": "auroc",
        "test/BinaryAveragePrecision/classification": "average_precision",
        "test/BinaryF1Score/classification": "f1_score",
        "test/BinaryCohenKappa/classification": "cohen_kappa",
        "test/BinaryRecall/classification": "recall",
        # --- Mantis Metrics ----
        "test/acc": "accuracy",
        "test/auroc": "auroc",
        "test/auprc": "average_precision",
        "test/f1": "f1_score",
        "test/cohen_kappa": "cohen_kappa",
        "test/recall": "recall",
    }

    df["metric"] = df["metric"].map(name_mapping).fillna(df["metric"])
    return df[["metric", "value"]]


def filter_datasets_by_class_count(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df[~df["dataset"].isin(label_10_filter)]
    return filtered_df


def main():
    # Define the path to the directory containing the results
    results_dir = Path("/data/stympopper/TSPFN-Benchmark/UCRUnivariate")

    # Initialize a list to store the results
    results = []

    # Iterate over each dataset directory
    for dataset_dir in tqdm(results_dir.iterdir(), desc="Processing datasets"):
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name.split("-TSPFN")[0]  # Extract dataset name

            # Iterate over each seed directory within the dataset directory
            for seed_dir in dataset_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                    seed_value = seed_dir.name.split("seed")[1]  # Extract seed value

                    # Define the path to the metrics file (assuming it's named "test_metrics.csv")
                    metrics_file = seed_dir / "test_metrics.csv"

                    if metrics_file.exists():
                        # Read the metrics file (assuming it's a CSV)
                        df = pd.read_csv(metrics_file)
                        df = simplify_csv_pandas(df)

                        # Extract relevant metrics (e.g., accuracy, F1-score)
                        accuracy = df.loc[df["metric"] == "accuracy", "value"].values[0]
                        f1_score = df.loc[df["metric"] == "f1_score", "value"].values[0]

                        # Append the results to the list
                        results.append(
                            {"dataset": dataset_name, "seed": seed_value, "accuracy": accuracy, "f1_score": f1_score}
                        )
                    else:
                        print(f"Metrics file not found for {dataset_name} with seed {seed_value}")

    # Convert results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results).sort_values(by=["dataset", "seed"])
    # results_df = filter_datasets_by_class_count(results_df)
    output_dir = "/data/stympopper/TSPFN-Benchmark"
    results_df.to_csv(f"{output_dir}/less_10_ucr_univariate_results_summary-CubePFN3-contrastive.csv", index=False)
    print("Results summary saved to less_10_ucr_univariate_results_summary-CubePFN3-contrastive.csv")

def main_v3():
    # Define the path to the directory containing the results
    results_dir = Path("/data/stympopper/TSPFN-Benchmark/UCRUnivariate").glob("*-TabPFN3/")

    # Initialize a list to store the results
    results = []

    # Iterate over each dataset directory
    for dataset_dir in tqdm(results_dir, desc="Processing datasets"):
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name.split("-TSPFN")[0]  # Extract dataset name

            # Iterate over each seed directory within the dataset directory
            for seed_dir in dataset_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                    seed_value = seed_dir.name.split("seed")[1]  # Extract seed value

                    # Define the path to the metrics file (assuming it's named "test_metrics.csv")
                    metrics_file = seed_dir / "test_metrics.csv"

                    if metrics_file.exists():
                        # Read the metrics file (assuming it's a CSV)
                        df = pd.read_csv(metrics_file)
                        df = simplify_csv_pandas(df)

                        # Extract relevant metrics (e.g., accuracy, F1-score)
                        accuracy = df.loc[df["metric"] == "accuracy", "value"].values[0]
                        f1_score = df.loc[df["metric"] == "f1_score", "value"].values[0]

                        # Append the results to the list
                        results.append(
                            {"dataset": dataset_name, "seed": seed_value, "accuracy": accuracy, "f1_score": f1_score}
                        )
                    else:
                        print(f"Metrics file not found for {dataset_name} with seed {seed_value}")

    # Convert results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results).sort_values(by=["dataset", "seed"])
    # results_df = filter_datasets_by_class_count(results_df)
    output_dir = "/data/stympopper/TSPFN-Benchmark"
    results_df.to_csv(f"{output_dir}/less_10_ucr_univariate_results_summary-TabPFN3/.csv", index=False)
    print("Results summary saved to less_10_ucr_univariate_results_summary-TabPFN3/.csv")

def main_mantis():
    # Define the path to the directory containing the results
    results_dir = Path("/data/stympopper/TSPFN-Benchmark/UCRUnivariate").glob("*-MantisV2-FineTune*")

    # Initialize a list to store the results
    results = []

    # Iterate over each dataset directory
    for dataset_dir in tqdm(results_dir, desc="Processing datasets"):
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name.split("-MantisV2-FineTune")[0]  # Extract dataset name

            # Iterate over each seed directory within the dataset directory
            for seed_dir in dataset_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                    seed_value = seed_dir.name.split("seed")[1]  # Extract seed value

                    # Define the path to the metrics file (assuming it's named "test_metrics.csv")
                    metrics_file = seed_dir / "mantis_rf_test_metrics.csv"

                    if metrics_file.exists():
                        # Read the metrics file (assuming it's a CSV)
                        df = pd.read_csv(metrics_file)
                        df = simplify_csv_pandas(df)

                        # Extract relevant metrics (e.g., accuracy, F1-score)
                        accuracy = df.loc[df["metric"] == "accuracy", "value"].values[0]
                        f1_score = df.loc[df["metric"] == "f1_score", "value"].values[0]

                        # Append the results to the list
                        results.append(
                            {"dataset": dataset_name, "seed": seed_value, "accuracy": accuracy, "f1_score": f1_score}
                        )
                    else:
                        print(f"Metrics file not found for {dataset_name} with seed {seed_value}")

    # Convert results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results).sort_values(by=["dataset", "seed"])
    # results_df = filter_datasets_by_class_count(results_df)
    output_dir = "/data/stympopper/TSPFN-Benchmark"
    results_df.to_csv(f"{output_dir}/less_10_ucr_univariate_results_summary-MantisV2-FineTune.csv", index=False)
    print("Results summary saved to less_10_ucr_univariate_results_summary-MantisV2-FineTune.csv")


if __name__ == "__main__":
    main_mantis()