import os
import numpy as np
import pandas as pd
import torch
import click
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--input_csv", type=str, required=True, help="Path to the input CSV file containing the dataset.")
@click.option("--output_dir", type=str, required=True, help="Directory where the split datasets will be saved.")
@click.option("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
@click.option("--random_state", type=int, default=42, help="Random seed for reproducibility.")
def split_dataset(input_csv: str, output_dir: str, test_size: float, random_state: int):
    # Load the dataset from the CSV file
    df = pd.read_csv(input_csv)
    labels = df.iloc[:, -1].values  # Assuming the last column contains labels
    features = df.iloc[:, :-1].values  # All columns except the last one are features

    # Check if the dataset is empty
    if df.empty:
        print("The input dataset is empty. Please provide a valid CSV file.")
        return

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=labels)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets to new CSV files
    train_output_path = os.path.join(output_dir, "train.csv")
    test_output_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Dataset successfully split and saved to {output_dir}")

if __name__ == "__main__":
    split_dataset()
