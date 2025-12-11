import argparse
from pathlib import Path

from .viewer import launch_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize paired 3D TIFF volumes and labels in napari."
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="Folder containing 3D training TIFF volumes (uint8).",
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        type=Path,
        help="Folder containing binary label TIFF volumes (uint8) with matching filenames.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        help="Optional path to a CSV file where flagged sample IDs will be appended (unique).",
    )
    args = parser.parse_args()

    launch_viewer(str(args.train_dir), str(args.label_dir), log_csv=args.log_csv)


if __name__ == "__main__":
    main()
