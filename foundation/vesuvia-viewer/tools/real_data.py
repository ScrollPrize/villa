import argparse
import asyncio
import json
import math
from pathlib import Path

import aiohttp
import numpy as np
import tifffile
import zarr

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4"
CT_URL = f"{BUCKET}/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
INK_URL = f"{BUCKET}/representations/predictions/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr"
SURFACE_URL = f"{BUCKET}/segments/20231221180251/mesh/20231221180251-on-20260411134726-2.4um.tifxyz"
FOCUS_ZYX = [44193.53, 13822.27, 20409.01]
SURFACE_FILES = ["meta.json", "x.tif", "y.tif", "z.tif"]
CONNECTIONS = 16
POOLING_TOLERANCE = 2.0        # mean |level k+1 − mean of level k| in uint8 steps; the scrolls measure about 0.3
SAMPLE = 3

parser = argparse.ArgumentParser(description="Cut of a real scroll from S3 into real/")
parser.add_argument("--out", required=True)
args = parser.parse_args()
out = Path(args.out)


async def fetch(session, url, path):
    async with session.get(url) as response:
        if response.status == 404:
            return False
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(await response.read())
        return True


async def pyramid(session, url, folder):
    attrs = folder / ".zattrs"
    await fetch(session, f"{url}/.zattrs", attrs)
    datasets = json.loads(attrs.read_text())["multiscales"][0]["datasets"]
    levels = []
    for dataset in datasets:
        level = dataset["path"]
        scale = dataset["coordinateTransformations"][0]["scale"]
        await fetch(session, f"{url}/{level}/.zarray", folder / level / ".zarray")
        array = json.loads((folder / level / ".zarray").read_text())
        d = [round(s) for s in scale]
        chunk = [math.floor(f / (d[a] * array["chunks"][a])) for a, f in enumerate(FOCUS_ZYX)]
        separator = array.get("dimension_separator", ".")
        key = separator.join(str(c) for c in chunk)
        held = await fetch(session, f"{url}/{level}/{key}", folder / level / Path(*key.split("/")))
        levels.append({"level": int(level), "downsample": d, "chunk": chunk, "held": held, "chunks": array["chunks"]})
        print(f"{folder.name} level {level}: downsample {d}, chunk {chunk} {'held' if held else 'is fill'}", flush=True)
    return levels


async def download():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONNECTIONS)) as session:
        surface = [fetch(session, f"{SURFACE_URL}/{name}", out / "surface.tifxyz" / name) for name in SURFACE_FILES]
        return (await asyncio.gather(pyramid(session, CT_URL, out / "ct.zarr"), pyramid(session, INK_URL, out / "ink.zarr"), *surface))[:2]


ct_levels, ink_levels = asyncio.run(download())


def arrays(folder, levels):
    return [zarr.open(str(folder / str(level["level"])), mode="r") for level in levels]


ct = arrays(out / "ct.zarr", ct_levels)
ink = arrays(out / "ink.zarr", ink_levels)


def pooling(name, opened, levels):
    for k in range(len(levels) - 1):
        d0, d1 = levels[k]["downsample"], levels[k + 1]["downsample"]
        if not (levels[k]["held"] and levels[k + 1]["held"]) or any(d1[a] % d0[a] for a in range(3)):
            continue
        ratio = [d1[a] // d0[a] for a in range(3)]
        start = [levels[k]["chunk"][a] * levels[k]["chunks"][a] for a in range(3)]
        size = [levels[k]["chunks"][a] for a in range(3)]
        fine = opened[k][tuple(slice(start[a], start[a] + size[a]) for a in range(3))].astype(np.float64)
        pooled = fine.reshape(size[0] // ratio[0], ratio[0], size[1] // ratio[1], ratio[1], size[2] // ratio[2], ratio[2]).mean(axis=(1, 3, 5))
        coarse = opened[k + 1][tuple(slice(start[a] // ratio[a], (start[a] + size[a]) // ratio[a]) for a in range(3))].astype(np.float64)
        difference = float(np.abs(coarse - pooled).mean())
        correlation = float(np.corrcoef(coarse.ravel(), pooled.ravel())[0, 1]) if coarse.std() > 0 and pooled.std() > 0 else 1.0
        print(f"{name} level {k + 1} against the mean of level {k}: mean |difference| {difference:.3f}, correlation {correlation:.4f}", flush=True)
        if difference > POOLING_TOLERANCE:
            raise SystemExit(f"{name} level {k + 1} is not the mean of level {k}: mean |difference| {difference:.3f}")


def probes(opened, levels):
    rows = []
    for k, level in enumerate(levels):
        if not level["held"]:
            continue
        d = level["downsample"]
        voxel = [math.floor(FOCUS_ZYX[a] / d[a]) for a in range(3)]
        chunk = opened[k][tuple(slice(level["chunk"][a] * level["chunks"][a], (level["chunk"][a] + 1) * level["chunks"][a]) for a in range(3))]
        rows.append({"level": k, "voxel": voxel, "world": [voxel[a] * d[a] + (d[a] - 1) / 2 for a in range(3)],
                     "value": int(opened[k][tuple(voxel)]), "chunkMean": float(chunk.mean())})
    return rows


x, y, z = (tifffile.imread(out / "surface.tifxyz" / f"{name}.tif").astype(np.float64) for name in "xyz")
squared = (z - FOCUS_ZYX[0]) ** 2 + (y - FOCUS_ZYX[1]) ** 2 + (x - FOCUS_ZYX[2]) ** 2
squared[(x == -1) & (y == -1) & (z == -1)] = np.inf
v, u = np.unravel_index(int(np.argmin(squared)), squared.shape)
point = [float(z[v, u]), float(y[v, u]), float(x[v, u])]
surface = None
for k, level in enumerate(ct_levels):
    d = level["downsample"]
    voxel = [math.floor(point[a] / d[a]) for a in range(3)]
    inside = all(level["chunk"][a] * level["chunks"][a] + SAMPLE <= voxel[a] < (level["chunk"][a] + 1) * level["chunks"][a] - SAMPLE for a in range(3))
    if not (level["held"] and inside):
        continue
    box = ct[k][tuple(slice(voxel[a] - SAMPLE // 2, voxel[a] + SAMPLE // 2 + 1) for a in range(3))]
    chunk = ct[k][tuple(slice(level["chunk"][a] * level["chunks"][a], (level["chunk"][a] + 1) * level["chunks"][a]) for a in range(3))]
    value, chunk_mean = float(box.mean()), float(chunk.mean())
    print(f"surface point uv {[int(u), int(v)]} at {point}, {math.dist(point, FOCUS_ZYX):.1f} from the focus:"
          f" CT {value:.1f} at level {k}, whose chunk averages {chunk_mean:.1f}", flush=True)
    if value <= chunk_mean:
        raise SystemExit(f"the surface point lies on CT of {value:.1f}, no brighter than its chunk's {chunk_mean:.1f}")
    surface = {"uv": [int(u), int(v)], "zyx": point, "level": k, "value": value, "chunkMean": chunk_mean}
    break
if surface is None:
    raise SystemExit(f"no held chunk holds the surface point {point}")

pooling("ct", ct, ct_levels)
pooling("ink", ink, ink_levels)
(out / "probes.json").write_text(json.dumps({
    "focus": FOCUS_ZYX,
    "ct": probes(ct, ct_levels),
    "ink": probes(ink, ink_levels),
    "surface": surface,
}, indent=1))
held = sum(level["held"] for level in ct_levels + ink_levels)
size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
print(f"real cut written to {out}: {held} chunks over {len(ct_levels)} CT and {len(ink_levels)} ink levels, the surface, {size / 2**20:.0f} MiB")
