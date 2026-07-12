import os
from pathlib import Path
import shutil
from glob import glob


def check_scores(path: Path, pattern: str = "CubePFN3-FineTune"):
    """
    Check for missing scores in the given path.
    """
    missing_files = []
    for folder in os.listdir(path):
        if folder.endswith(pattern):
            print(f"Checking scores in {folder}")
            if not any(glob(os.path.join(path, folder, "**", "test_metrics.csv"), recursive=True)):
                missing_files.append(folder)
    if missing_files:
        print("Missing scores found in the following directories:")
        for missing in missing_files:
            print(missing.split("-")[0])  # Print only the dataset name

if __name__ == "__main__":
    path = "/data/stympopper/TSPFN-Benchmark/UCRUnivariate"
    check_scores(path, pattern="-TabPFN3-FineTune-NoScheduler")