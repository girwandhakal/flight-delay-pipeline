"""Run the BTS workflow in order, stopping when a stage fails."""

from pathlib import Path
import sys

# Find the project scripts even when this file is run from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bts_download
import bts_clean
import bts_transform

# Run acquisition, cleaning, feature preparation, training, and evaluation in order.
def main():
    if bts_download.main():
        return 1
    if bts_clean.main():
        return 1
    if bts_transform.main():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
