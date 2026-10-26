"""Shared locations; inputs and editable UI files remain beside the programs."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CAL_DIR = APP_DIR / "cal"
OUTPUT_DIR = APP_DIR / "output"


def output_dir(category):
    folder = OUTPUT_DIR / category
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def calibration_input(path):
    """Read legacy/restored root-level RTs without making them the save default."""
    path = Path(path)
    if path.parent.resolve() != (OUTPUT_DIR / "calibration").resolve():
        return path
    legacy = APP_DIR / path.name
    return legacy if not path.exists() and legacy.is_file() else path
