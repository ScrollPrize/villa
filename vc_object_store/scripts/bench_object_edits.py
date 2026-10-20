#!/usr/bin/env python3
"""Run the native edit persistence benchmark serially on real tifxyz inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--radius", type=float, action="append")
    parser.add_argument("--binary", type=Path, default=Path("build/bin/bench_object_edits"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--edits", type=int, default=1000)
    parser.add_argument("--file-saves", type=int, default=50)
    parser.add_argument("--per-point", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # Exclusive creation prevents accidentally replacing an earlier measurement.
    with args.output.open("x") as output, tempfile.TemporaryDirectory(prefix="vc-edit-bench-") as temp:
        for repeat in range(args.repeats):
            for source in args.segment:
                for radius in args.radius or [5.75, 16, 64]:
                    command = [
                        str(args.binary.resolve()), str(source.resolve()),
                        str(Path(temp) / "work"), str(radius), str(args.edits), str(args.file_saves),
                    ]
                    if args.per_point:
                        command.append("per-point")
                    run = subprocess.run(command, check=True, capture_output=True, text=True)
                    result = json.loads(run.stdout.splitlines()[-1])
                    result["repeat"] = repeat + 1
                    result["command"] = command
                    result["source_metadata"] = json.loads((source / "meta.json").read_text())
                    output.write(json.dumps(result) + "\n")
                    output.flush()
                    print(
                        f"{source.name} radius={radius} repeat={repeat + 1}: "
                        f"DB p50/p95={result['db_ms']['p50']:.3f}/{result['db_ms']['p95']:.3f} ms, "
                        f"file p50={result['file_warm_ms']['p50']:.3f} ms",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
