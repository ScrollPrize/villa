"""Format example for VC3D plugins: write a VC3D point collection around a point.

    python3 points_at_focus.py --point X,Y,Z --radius 20 --out points.json

The file uses VC3D's point collection format (vc_pointcollections_json_version 1),
the same one File > Save in the Point Collections panel writes. Standard library only.
"""

import argparse
import json
import math
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", required=True, help="x,y,z in full-resolution voxels")
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        x, y, z = (float(v) for v in args.point.split(","))
    except ValueError:
        parser.error("--point must be three numbers: x,y,z")
    if not all(math.isfinite(v) for v in (x, y, z, args.radius)) or args.radius < 0:
        parser.error("--point and --radius must be finite and the radius not negative")

    r = args.radius
    offsets = [(0, 0, 0), (r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
    points = {str(i + 1): {"p": [x + dx, y + dy, z + dz]} for i, (dx, dy, dz) in enumerate(offsets)}
    doc = {
        "vc_pointcollections_json_version": "1",
        "collections": {"1": {"name": "focus-cross", "points": points, "color": [1.0, 0.6, 0.1]}},
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {len(points)} points to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
