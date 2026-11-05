"""Locked reviewed-failure cohort helpers for training-time regression tests."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from .replay_metrics import reference_segments


MANIFEST_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def failed_piece_paths(summary: dict) -> set[Path]:
    """Return whole truncated pieces for which either direction restarted."""
    if not summary.get("completed") or summary.get("infrastructure_failures", 0):
        raise ValueError("canonical reviewed evaluation must be complete and infrastructure-clean")
    selected: set[Path] = set()
    for source in summary["sources"]:
        fibers = source["backend_config"]["fibers"]
        for case in source["cases"]:
            if len(reference_segments(case)) > 1:
                selected.add(Path(fibers[int(case["source_index"])]).resolve())
    return selected


def canonical_failure_union(summary_paths: list[Path]) -> set[Path]:
    if not summary_paths:
        raise ValueError("at least one canonical evaluation is required")
    selected: set[Path] = set()
    for path in summary_paths:
        selected.update(failed_piece_paths(json.loads(path.read_text())))
    return selected


def materialize_failure_union(summary_paths: list[Path], destination: Path) -> dict:
    """Copy the exact union into a runner-compatible locked split."""
    summary_paths = [Path(path).resolve() for path in summary_paths]
    selected = sorted(canonical_failure_union(summary_paths), key=str)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"refusing to replace nonempty failure test root: {destination}")
    rows = []
    for source in selected:
        try:
            collection = source.parent.name
        except IndexError as error:
            raise ValueError(f"cannot derive collection from {source}") from error
        target = destination / "test" / collection / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"source": collection, "split": "test",
                     "path": f"{collection}/{source.name}",
                     "sha256": sha256_file(target), "source_path": str(source)})
    manifest = {
        "version": MANIFEST_VERSION,
        "selection": "union-of-pieces-with-restart-in-either-direction",
        "canonical_summaries": [str(path) for path in summary_paths],
        "expected_count": len(rows),
        "files": rows,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_failure_union(destination, expected_count=len(rows))
    return manifest


def validate_failure_union(root: Path, *, expected_count: int = 32) -> dict:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported reviewed-failure manifest version")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != expected_count or manifest.get("expected_count") != expected_count:
        raise ValueError(f"reviewed-failure cohort must contain exactly {expected_count} pieces")
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate reviewed-failure manifest path")
    actual = {str(path.relative_to(root / "test")): path
              for path in sorted((root / "test").glob("*/*.json"))}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"reviewed-failure membership changed: missing={missing} extra={extra}")
    for name, path in actual.items():
        if sha256_file(path) != expected[name]["sha256"]:
            raise ValueError(f"reviewed-failure annotation hash changed: {name}")
    return manifest


def directional_restart_metrics(summary: dict, *, expected_pieces: int = 32) -> dict[str, float]:
    """Aggregate actual restart episodes and implied lengths by direction."""
    if not summary.get("completed") or summary.get("infrastructure_failures", 0):
        raise ValueError("trace result must be complete and infrastructure-clean")
    expected_cases = 2 * expected_pieces
    if int(summary.get("evaluated_cases", summary.get("cases", -1))) != expected_cases:
        raise ValueError(f"trace result must contain exactly {expected_cases} directed cases")
    totals = defaultdict(lambda: {"restarts": 0, "reference_mm": 0.0, "cases": 0})
    for source in summary["sources"]:
        scale = float(source["summary"]["base_voxel_size_um"]) / 1000.0
        for case in source["cases"]:
            direction = str(case["direction"])
            if direction not in ("forward", "reverse"):
                raise ValueError(f"unknown trace direction: {direction}")
            segments = reference_segments(case, scale)
            totals[direction]["restarts"] += len(segments) - 1
            totals[direction]["reference_mm"] += math.fsum(segments)
            totals[direction]["cases"] += 1
    if any(totals[d]["cases"] != expected_pieces for d in ("forward", "reverse")):
        raise ValueError("trace result must contain one forward and reverse case per piece")
    result: dict[str, float] = {}
    for direction in ("forward", "reverse"):
        row = totals[direction]
        result[f"restarts_{direction}"] = row["restarts"]
        result[f"reference_mm_{direction}"] = row["reference_mm"]
        result[f"implied_mean_mm_{direction}"] = row["reference_mm"] / (row["restarts"] + 1)
    restarts = totals["forward"]["restarts"] + totals["reverse"]["restarts"]
    reference = totals["forward"]["reference_mm"] + totals["reverse"]["reference_mm"]
    result.update(restarts_combined=restarts, reference_mm_combined=reference,
                  implied_mean_mm_combined=reference / (restarts + 1))
    return result
