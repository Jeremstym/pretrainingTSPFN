import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--input_csv", type=click.Path(exists=True), required=True, help="Path to the input CSV.")
@click.option("--output_dir", type=str, required=True, help="Directory to save the splits.")
@click.option("--test_size", type=float, default=0.2, help="Proportion of test data.")
@click.option("--seed", type=int, default=42, help="Random seed.")
def split_dataset(input_csv: str, output_dir: str, test_size: float, seed: int):
    """Splits a CSV into stratified train and test sets."""
    
    df = pd.read_csv(input_csv)
    
    if df.empty:
        click.echo("Error: The input dataset is empty.")
        return

    # Using -1 for labels is fine, but we ensure stratify gets the values correctly
    y = df.iloc[:, -1]
    
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=seed, 
            stratify=y
        )
    except ValueError as e:
        click.echo(f"Split failed: {e}. (Hint: Check if a class has only 1 member)")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    click.secho(f"🚀 Success! Files saved to: {output_dir}", fg="green")

if __name__ == "__main__":
    split_dataset()