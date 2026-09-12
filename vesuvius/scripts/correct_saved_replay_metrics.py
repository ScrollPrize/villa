#!/usr/bin/env python3
"""Correct saved replay-set summaries only. Default is a read-only dry run.

Example: PYTHONPATH=vesuvius/src python vesuvius/scripts/correct_saved_replay_metrics.py
    /ephemeral/hendrik/fiber3d_train2/evaluations --apply
Original bytes are backed up once; raw cases/trace.json files are never written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import correct_summary


def migrate(path: Path, apply: bool = False) -> dict:
    raw = path.read_bytes()
    original = json.loads(raw)
    corrected = correct_summary(original)
    old_hash = hashlib.sha256(raw).hexdigest()
    if corrected == original:
        return dict(path=str(path), status="already-correct", sha256=old_hash)
    backup = path.with_name("summary.before-complete-segments.json")
    if backup.exists() and backup.read_bytes() != raw:
        raise ValueError(f"refusing to overwrite a different migration backup: {backup}")
    corrected["metric_correction"] = dict(original_summary_sha256=old_hash,
                                        backup=backup.name, raw_traces_modified=False)
    new_raw = (json.dumps(corrected, indent=2, allow_nan=False) + "\n").encode()
    if apply:
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(raw)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metrics-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(new_raw)
            temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return dict(path=str(path), status="corrected" if apply else "would-correct",
                original_sha256=old_hash, corrected_sha256=hashlib.sha256(new_raw).hexdigest(),
                directions=corrected["cases"], failures=corrected["failures"],
                segments=corrected["reliability_segments"], mean_mm=corrected["mean_segment_length_mm"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    # Only evaluation metadata: no walk through cache, inference, or volume data.
    paths = [args.root] if args.root.is_file() else sorted(set(
        args.root.glob("*/summary.json")) | set(args.root.glob("*/*/summary.json")))
    # Validate every candidate before modifying any file.
    planned = [migrate(path) for path in paths]
    records = [migrate(path, True) for path in paths] if args.apply else planned
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
