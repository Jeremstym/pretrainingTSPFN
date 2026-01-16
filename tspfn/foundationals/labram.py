import sys
from pathlib import Path

# Add LaBraM to path before any labram-specific imports
labram_path = Path("submodules/labram").resolve()
sys.path.append(str(labram_path))
