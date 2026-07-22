#!/usr/bin/env python3
"""Download scale 0 for a list of public Vesuvius OME-Zarr volumes.

Each non-empty, non-comment line in the input file must be a full URI like::

    s3://vesuvius-challenge-open-data/PHerc0125/volumes/example.zarr

Volumes are processed one at a time. The underlying downloader transfers the
current volume's chunks concurrently (512 workers by default).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


BUCKET_URI = "s3://vesuvius-challenge-open-data/"


@dataclass(frozen=True)
class Volume:
    source: str
    scroll: str
    name: str

    def relative_path(self) -> Path:
        return Path(self.scroll) / "volumes" / self.name


def parse_volume_uri(raw: str, *, line_number: int) -> Volume:
    source = raw.strip().rstrip("/")
    if not source.startswith(BUCKET_URI):
        raise ValueError(
            f"line {line_number}: expected URI below {BUCKET_URI}, got {raw!r}"
        )

    relative = PurePosixPath(source.removeprefix(BUCKET_URI))
    parts = relative.parts
    if len(parts) != 3 or parts[1] != "volumes":
        raise ValueError(
            f"line {line_number}: expected <scroll>/volumes/<name>.zarr, got {raw!r}"
        )
    scroll, _volumes, name = parts
    if not scroll.startswith("PHerc") or not scroll.replace("_", "").isalnum():
        raise ValueError(f"line {line_number}: invalid scroll directory {scroll!r}")
    if not name.endswith(".zarr") or name in {".zarr", "..zarr"}:
        raise ValueError(f"line {line_number}: invalid volume directory {name!r}")
    return Volume(source=source, scroll=scroll, name=name)


def read_volume_list(path: Path) -> list[Volume]:
    volumes: list[Volume] = []
    seen_sources: set[str] = set()
    seen_scrolls: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        volume = parse_volume_uri(stripped, line_number=line_number)
        if volume.source in seen_sources:
            raise ValueError(f"line {line_number}: duplicate volume URI {volume.source}")
        if volume.scroll in seen_scrolls:
            raise ValueError(
                f"line {line_number}: multiple volumes for {volume.scroll} would overwrite info.json"
            )
        seen_sources.add(volume.source)
        seen_scrolls.add(volume.scroll)
        volumes.append(volume)
    if not volumes:
        raise ValueError(f"volume list is empty: {path}")
    return volumes


def write_info(scroll_dir: Path, volume: Volume) -> None:
    payload = {
        "s3_path": volume.source,
        "local_volume_path": (Path("volumes") / volume.name).as_posix(),
    }
    info_path = scroll_dir / "info.json"
    tmp_path = scroll_dir / ".info.json.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(info_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download scale 0 of each S3 OME-Zarr URI in a text file."
    )
    parser.add_argument("volume_list", type=Path, help="One full S3 volume URI per line")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help="Directory that will contain PHerc####/volumes/ (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=512,
        help="Parallel chunk transfers within each volume (default: 512)",
    )
    parser.add_argument(
        "--downloader",
        type=Path,
        default=Path(__file__).with_name("download_omezarr.py"),
        help="Path to download_omezarr.py",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later volumes after a downloader failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and destinations without creating files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    try:
        volumes = read_volume_list(args.volume_list)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    downloader = args.downloader.resolve()
    if not downloader.is_file():
        raise SystemExit(f"downloader script not found: {downloader}")
    output_root = args.output_root.resolve()
    failures = 0

    for index, volume in enumerate(volumes, 1):
        destination = output_root / volume.relative_path()
        scroll_dir = output_root / volume.scroll
        command = [
            sys.executable,
            str(downloader),
            volume.source,
            str(destination),
            "--scales",
            "0",
            "--anon",
            "--workers",
            str(args.workers),
        ]
        print(f"[{index}/{len(volumes)}] {volume.source}", flush=True)
        print(f"  -> {destination}", flush=True)
        print("  $ " + shlex.join(command), flush=True)
        if args.dry_run:
            continue

        scroll_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            write_info(scroll_dir, volume)
            continue

        failures += 1
        print(
            f"download failed for {volume.source} (exit {result.returncode})",
            file=sys.stderr,
            flush=True,
        )
        if not args.keep_going:
            return result.returncode or 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
