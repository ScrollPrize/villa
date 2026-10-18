"""Re-encode a real 2D TIFF as PNG, JPEG, and tiled Deflate TIFF for benchmarks.

JPEG samples are intentionally lossy; benchmark comparisons use its own decoded
values. The manifest records the original and derivative hashes explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'fixtures/fiber_132487/mask.tif')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = tifffile.imread(args.input)
    if data.ndim != 2 or data.dtype.name != 'uint8':
        raise ValueError('format benchmark preparation expects a uint8 YX TIFF')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir/name for name in ('mask.png', 'mask.jpg', 'mask-deflate.tif')]
    Image.fromarray(data).save(paths[0])
    Image.fromarray(data).save(paths[1], quality=95)
    tifffile.imwrite(paths[2], data, compression='deflate', tile=(16, 128))
    manifest = dict(input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    shape=data.shape, dtype=str(data.dtype), derivatives={p.name: dict(
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in paths})
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
