"""Non-CI driver for timing representative native patch-merger datasets."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/bin/merge_overlapping_patches"),
    )
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument(
        "--expected-patches",
        type=int,
        default=80_000,
        help="Warn if discovery does not match the representative 80k-patch run.",
    )
    args = parser.parse_args()
    completed = subprocess.run(
        [
            str(args.executable),
            str(args.patch_dir),
            str(args.output_dir),
            "--threads",
            str(args.threads),
            "--benchmark",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["input_count"] != args.expected_patches:
        print(
            f"warning: expected {args.expected_patches:,} patches, "
            f"discovered {report['input_count']:,}"
        )
    if report["total_duration"] > 300:
        print("warning: run exceeded the five-minute target")


if __name__ == "__main__":
    main()
