#!/usr/bin/env python
"""
Propose acquisition blocks for a flat folder of turntable photographs without annotation shots.

`scan` reads capture time and camera settings with ExifTool, splits the photographs into bursts
wherever the time between consecutive frames exceeds --burst-gap or a setting (zoom, exposure,
aperture, ISO, size) changes, and joins consecutive bursts with identical settings separated by
less than --block-gap into one block (typically two turntable turns). It reports frame-number
holes inside blocks and byte-identical duplicate files (e.g. DSC01234_1.ARW from a repeated
copy), which are listed in duplicates.txt and left out of the blocks. It writes:
  - blocks.tsv: one row per proposed block (name, first and last file, count, settings, gap
    before it). Rename the blocks to <orientation>_camera_<height> after inspecting the sheet,
    and merge or split rows where the timing misled the grouping;
  - contact.jpg: one row per burst with evenly spaced embedded previews, labelled by block.
Timing alone cannot tell a pause within a block from a short break between two blocks at the
same zoom, so always confirm the blocks and orientation changes on the contact sheet.

`link` builds TREE_DIR/<name>/ directories of symlinks to the files of each block in
blocks.tsv (skipping those in duplicates.txt beside it), the layout sam2-photogrammetry expects.
Camera JPEGs (captures without RAW files) are accepted too: there is nothing for raw2jpg.py to
convert, so `link` hard-links them straight to TREE_DIR/<name>/JPGEnhanced/<stem>.jpg, ready for
segment.py. macOS AppleDouble files (._DSC01234.JPG, left by copying from a Mac) are ignored.

Usage:
    python find_blocks.py scan RAW_DIR OUT_DIR [--burst-gap 5] [--block-gap 30]
    python find_blocks.py link OUT_DIR/blocks.tsv RAW_DIR TREE_DIR
"""

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
import re
import subprocess

from PIL import Image, ImageDraw

RAW_EXT = (".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2")
JPEG_EXT = (".jpg", ".jpeg")
SETTINGS = ("FocalLength", "ExposureTime", "FNumber", "ISO", "ImageSize")
TILE_W, TILE_H, LABEL_W, COLS = 240, 160, 150, 8


def photographs(raw_dir):
    """RAW and camera-JPEG files of a capture folder, without macOS AppleDouble ._ files."""
    return sorted(f for f in os.listdir(raw_dir)
                  if f.lower().endswith(RAW_EXT + JPEG_EXT) and not f.startswith("._"))


def link_photo(src, block_dir):
    """Symlink a RAW file into the block directory; hard-link a camera JPEG as JPGEnhanced/<stem>.jpg."""
    name = os.path.basename(src)
    if not name.lower().endswith(JPEG_EXT):
        os.symlink(os.path.abspath(src), os.path.join(block_dir, name))
        return
    jdir = os.path.join(block_dir, "JPGEnhanced")
    os.makedirs(jdir, exist_ok=True)
    os.link(src, os.path.join(jdir, os.path.splitext(name)[0] + ".jpg"))  # same filesystem, as for the PGS input


def read_exif(raw_dir):
    files = photographs(raw_dir)
    if not files:
        raise SystemExit(f"no RAW or JPEG files in {raw_dir}")
    out = subprocess.run(["exiftool", "-j", "-n", "-DateTimeOriginal", *("-" + s for s in SETTINGS), *files],
                         cwd=raw_dir, check=True, capture_output=True, text=True).stdout
    rows = sorted(json.loads(out), key=lambda r: r["SourceFile"])
    for r in rows:
        r["time"] = datetime.datetime.strptime(r["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
        r["settings"] = tuple(r.get(s) for s in SETTINGS)
    return rows


def find_duplicates(raw_dir, rows):
    """Files whose bytes equal an earlier file with the same capture time."""
    dup = set()
    by_time = collections.defaultdict(list)
    for r in rows:
        by_time[r["time"]].append(r["SourceFile"])
    for names in by_time.values():
        seen = {}
        for n in names:
            with open(os.path.join(raw_dir, n), "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h in seen:
                dup.add(n)
                print(f"duplicate: {n} is identical to {seen[h]}")
            else:
                seen[h] = n
    return dup


def split(rows, burst_gap, block_gap):
    """Return bursts (lists of rows) and the block index of each burst."""
    bursts = [[rows[0]]]
    for prev, r in zip(rows, rows[1:]):
        if (r["time"] - prev["time"]).total_seconds() > burst_gap or r["settings"] != prev["settings"]:
            bursts.append([r])
        else:
            bursts[-1].append(r)
    block_of = [0]
    for prev, b in zip(bursts, bursts[1:]):
        gap = (b[0]["time"] - prev[-1]["time"]).total_seconds()
        same = b[0]["settings"] == prev[-1]["settings"]
        block_of.append(block_of[-1] + (0 if same and gap <= block_gap else 1))
    return bursts, block_of


def frame_number(name):
    m = re.search(r"\d+", name)  # camera counter; copy suffixes like _1 come after it
    return int(m.group(0)) if m else None


def contact_sheet(raw_dir, bursts, block_of, out_path):
    sheet = Image.new("RGB", (COLS * TILE_W + LABEL_W, len(bursts) * TILE_H), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (burst, block) in enumerate(zip(bursts, block_of)):
        draw.text((4, row * TILE_H + 50), f"block{block + 1:02d}\n{burst[0]['SourceFile']}\n..{burst[-1]['SourceFile']}\n"
                  f"n={len(burst)} f={burst[0]['FocalLength']}", fill="black")
        for col in range(COLS):
            name = burst[(col * len(burst)) // COLS]["SourceFile"]
            preview = subprocess.run(["exiftool", "-b", "-PreviewImage", name], cwd=raw_dir, capture_output=True).stdout
            if not preview:
                continue
            tile = Image.open(io.BytesIO(preview))
            tile.thumbnail((TILE_W - 2, TILE_H - 2))
            sheet.paste(tile, (LABEL_W + col * TILE_W, row * TILE_H))
    sheet.save(out_path, quality=85)
    print("wrote", out_path)


def scan(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_exif(args.raw_dir)
    dup = find_duplicates(args.raw_dir, rows)
    with open(os.path.join(args.out_dir, "duplicates.txt"), "w") as f:
        f.writelines(n + "\n" for n in sorted(dup))
    rows = [r for r in rows if r["SourceFile"] not in dup]
    bursts, block_of = split(rows, args.burst_gap, args.block_gap)
    print(f"{len(rows)} photographs ({len(dup)} duplicates left out), {len(bursts)} bursts, {block_of[-1] + 1} blocks")
    blocks = collections.defaultdict(list)
    for burst, b in zip(bursts, block_of):
        blocks[b].append(burst)
    with open(os.path.join(args.out_dir, "blocks.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["name", "first", "last", "count", *SETTINGS, "gap_before_s"])
        prev_end = None
        for b, bb in sorted(blocks.items()):
            frames = [r for burst in bb for r in burst]
            gap = "" if prev_end is None else int((frames[0]["time"] - prev_end).total_seconds())
            prev_end = frames[-1]["time"]
            w.writerow([f"block{b + 1:02d}", frames[0]["SourceFile"], frames[-1]["SourceFile"], len(frames),
                        *frames[0]["settings"], gap])
            nums = [frame_number(r["SourceFile"]) for r in frames]
            holes = sorted(set(range(min(nums), max(nums) + 1)) - set(nums)) if None not in nums else []
            within = [int((y[0]["time"] - x[-1]["time"]).total_seconds()) for x, y in zip(bb, bb[1:])]
            print(f"block{b + 1:02d}: {frames[0]['SourceFile']} .. {frames[-1]['SourceFile']}  n={len(frames):3d}  "
                  f"focal {frames[0]['FocalLength']}  turns/bursts {[len(x) for x in bb]}  gaps within {within} s  "
                  + (f"gap before {gap} s" if gap != "" else "first block")
                  + (f"  missing frame numbers {holes}" if holes else ""))
    print("wrote", os.path.join(args.out_dir, "blocks.tsv"))
    contact_sheet(args.raw_dir, bursts, block_of, os.path.join(args.out_dir, "contact.jpg"))


def link(args):
    dup_path = os.path.join(os.path.dirname(os.path.abspath(args.blocks_tsv)), "duplicates.txt")
    dup = set(open(dup_path).read().split()) if os.path.exists(dup_path) else set()
    names = [r["SourceFile"] for r in read_exif(args.raw_dir) if r["SourceFile"] not in dup]
    with open(args.blocks_tsv) as f:
        blocks = list(csv.DictReader(f, delimiter="\t"))
    total = 0
    for b in blocks:
        i, j = names.index(b["first"]), names.index(b["last"])
        members = names[i:j + 1]
        d = os.path.join(args.tree_dir, b["name"])
        os.makedirs(d)
        for n in members:
            link_photo(os.path.join(args.raw_dir, n), d)
        total += len(members)
        if len(members) != int(b["count"]):
            print(f"warning: {b['name']} has {len(members)} files between {b['first']} and {b['last']}, "
                  f"blocks.tsv says {b['count']} (duplicates or edited rows?)")
        print(f"{b['name']}: {len(members)} files")
    print(f"linked {total} of {len(names)} files" + (f" ({len(dup)} duplicates skipped)" if dup else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan", help="propose blocks and write blocks.tsv and contact.jpg")
    p.add_argument("raw_dir")
    p.add_argument("out_dir")
    p.add_argument("--burst-gap", type=float, default=5, help="seconds between frames that start a new burst")
    p.add_argument("--block-gap", type=float, default=30, help="longest pause between bursts of one block (s)")
    p = sub.add_parser("link", help="build the block directory tree from blocks.tsv")
    p.add_argument("blocks_tsv")
    p.add_argument("raw_dir")
    p.add_argument("tree_dir")
    args = parser.parse_args()
    scan(args) if args.command == "scan" else link(args)


if __name__ == "__main__":
    main()
