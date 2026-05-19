import os
import sys
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    # Define the path to the directory containing the results
    results_dir = glob("/data/stympopper/TSPFN-Benchmark/UCRUnivariate/*/seed42/test_metrics.csv")

    # Initialize a list to store the results
    results = []

    # Iterate over each dataset directory
    for dataset_dir in tqdm(results_dir, desc="Processing datasets"):
        dataset_name = dataset_dir.name.split("-TSPFN")[0]  # Extract dataset name
        print(f"Processing dataset: {dataset_name}")
        raise NotImplementedError("This script is not yet implemented. Please implement the logic to read the metrics files and extract the relevant metrics (e.g., accuracy, F1-score) for each dataset and seed. Then, store the results in a structured format (e.g., a list of dictionaries) and save it to a CSV file for further analysis.")

        # Iterate over each seed directory within the dataset directory
        for seed_dir in dataset_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                seed_value = seed_dir.name.split("seed")[1]  # Extract seed value

                # Define the path to the metrics file (assuming it's named "metrics.csv")
                metrics_file = seed_dir / "metrics.csv"

                if metrics_file.exists():
                    # Read the metrics file (assuming it's a CSV)
                    df = pd.read_csv(metrics_file)

                    # Extract relevant metrics (e.g., accuracy, F1-score)
                    accuracy = df.loc[df['metric'] == 'accuracy', 'value'].values[0]
                    f1_score = df.loc[df['metric'] == 'f1_score', 'value'].values[0]

                    # Append the results to the list
                    results.append({
                        "dataset": dataset_name,
                        "seed": seed_value,
                        "accuracy": accuracy,
                        "f1_score": f1_score
                    })
                else:
                    print(f"Metrics file not found for {dataset_name} with seed {seed_value}")

    # Convert results to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("ucr_univariate_results_summary.csv", index=False)
    print("Results summary saved to ucr_univariate_results_summary.csv")


if __name__ == "__main__":
    main()