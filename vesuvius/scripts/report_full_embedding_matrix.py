#!/usr/bin/env python3
"""Read-only report of the locked full-cohort embedding matrix and history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vesuvius/src"), str(ROOT / "vesuvius/scripts"), str(ROOT)]
from fiber3d_comparison_report import comparison_markdown, classic_normalized_score
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import complete_report

DEFAULT = ROOT / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/embedding_full_matrix_20260914_manifest.json"


def read_row(name, output, *, new=False, config=None):
    folder = Path(output) / "full"
    if any((folder / f).exists() for f in ("incomplete.json", "inference_incomplete.json")):
        raise ValueError(f"incomplete evaluation: {folder}")
    raw = (folder / "summary.json").read_bytes()
    summary = json.loads(raw)
    if not summary.get("completed") or summary.get("cases") != 100:
        raise ValueError(f"expected100 completed directions: {folder}")
    row = dict(name=name, output=str(output), metrics=complete_report(summary),
               summary_sha256=hashlib.sha256(raw).hexdigest(), new=new)
    selection = Path(output) / "selection.json"
    if selection.exists():
        row["selection"] = json.loads(selection.read_text())
    if new:
        if len(list((folder / "cases").glob("*/trace.json"))) != 100:
            raise ValueError(f"missing raw case artifacts: {folder}")
        row["config"] = config
        row["effective_trace"] = summary.get("effective_trace")
        from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import distribution
        inference_raw = (folder / "inference_summary.json").read_bytes()
        inference = json.loads(inference_raw)
        pool = inference["worker_pool"]
        row["inference_summary_sha256"] = hashlib.sha256(inference_raw).hexdigest()
        row["inference"] = dict(
            coordinator_seconds=inference["wall_seconds"],
            native_seconds=summary.get("wall_seconds"),
            sources=inference["sources"],
            batch_seconds=distribution(pool["batch_seconds"]),
            worker_pool={k: v for k, v in pool.items() if k != "batch_seconds"})
        row["case_artifacts"] = str(folder / "cases")
    return row


def report(manifest, *, allow_partial=False):
    rows = [read_row(r["name"], r["output"]) for r in manifest["historical"]]
    pending = []
    for path in manifest["configs"]:
        config = json.loads((ROOT / path).read_text())
        name = Path(path).stem.removeprefix("eval_embedding_full_").split("_20260914_", 1)[0]
        output = config["output_root"]
        if not (Path(output) / "full/summary.json").exists():
            if not allow_partial:
                raise ValueError(f"evaluation still pending: {name}")
            pending.append(name)
            continue
        row = read_row("Embedding EMA99 " + name, output, new=True, config=config)
        metadata = manifest["snapshots"][name.rsplit("_", 1)[0]]
        if metadata["sha256"] != config["checkpoint_sha256"]:
            raise ValueError("snapshot provenance differs from configuration")
        row["selection"] = dict(sha256=metadata["sha256"], run=metadata["run"],
            selected=dict(path=metadata["original_path"], step=metadata["step"],
                          samples=metadata["samples"], metric=metadata["metric"],
                          metric_name=metadata["metric_name"], decay=0.99))
        rows.append(row)
    markdown = comparison_markdown(rows)  # Shared coverage and metric audits.
    classic = next(r["metrics"] for r in rows if r["name"] == "Classic")
    for row in rows:
        row["classic_normalized_score"] = classic_normalized_score(row["metrics"], classic)
    if pending:
        markdown = "# INCOMPLETE MATRIX\n\nPending: " + ", ".join(pending) + "\n\n" + markdown
    markdown += "\n## Execution caveats\n\n"
    markdown += ("New runs use eight shared20GiB GPU grants,96CPU,batch1,float32 accumulation; "
                 "historical full runs generally used exclusive8GPU,batch8,float16 accumulation. "
                 "No isolated runtime comparison or bit-identical inference claim. "
                 "The hard tuning fiber is included in the full cohort. "
                 "All actual failure positions and trace geometry remain in each new run's full/cases directory.\n")
    return dict(complete=not pending, pending=pending, manifest=manifest, rows=rows), markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    data, markdown = report(json.loads(args.manifest.read_text()), allow_partial=args.allow_partial)
    print(json.dumps(data, indent=2) if args.format == "json" else markdown)


if __name__ == "__main__":
    main()
