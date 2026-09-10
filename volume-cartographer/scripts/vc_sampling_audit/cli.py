"""Audit final renderer rays; between-pixel interpolation is a surrogate."""
import argparse
import json
from pathlib import Path

from .capture import report_capture
from .provenance import code_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--reference", type=float, nargs=3, required=True,
                   help="Declared oriented chart reference XYZ; not a fitted normal correction")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Fresh additive output required")
    result = report_capture(args.capture, args.reference)
    result["code_sha256"] = code_sha256()
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "report": str(args.out),
                      "depth": [float(x) for x in result["diagonals"]["AC"]["depth_interval_input_units"]]}))
