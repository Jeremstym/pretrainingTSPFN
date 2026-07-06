import os
from pathlib import Path
import shutil
from glob import glob


def remove_checkpoints_directory(path: Path, pattern: str = "-FineTune"):
    """
    Remove all checkpoint directories in the given path.
    """
    for folder in os.listdir(path):
        if folder.endswith(pattern):
            print(f"Removing checkpoints in {folder}")
            for checkpoint_file in glob(os.path.join(path, folder, "**", "checkpoints")):
                shutil.rmtree(checkpoint_file)
    # for checkpoint_file in glob(os.path.join(path, "**", "checkpoints"), recursive=True):
    #     shutil.rmtree(checkpoint_file)

if __name__ == "__main__":
    path = "/data/stympopper/TSPFN-Benchmark/UCRUnivariate"
    remove_checkpoints_directory(path, pattern="-FineTune")