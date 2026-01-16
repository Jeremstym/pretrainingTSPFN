import sys
from pathlib import Path

# This finds the root directory relative to this __init__ file
# tspfn/__init__.py -> tspfn/ -> project_root/
PROJECT_ROOT = Path(__file__).parent.parent
LABRAM_PATH = PROJECT_ROOT / "submodules" / "labram"

if LABRAM_PATH.exists():
    if str(LABRAM_PATH) not in sys.path:
        sys.path.append(str(LABRAM_PATH))
else:
    print(f"Warning: LaBraM submodule not found at {LABRAM_PATH}")