"""One frozen-current-checkpoint, one-fiber off/on live tracing comparison."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time


def select_pilot_fiber(source, requested, length):
    if requested is None:
        return min(source["fibers"], key=lambda path: (length(path), path))
    matches = [path for path in source["fibers"] if Path(path).resolve() == Path(requested).resolve()]
    if len(matches) != 1:
        raise ValueError("pilot fiber must match exactly one prepared source fiber")
    length(matches[0])  # Validate the same single-fiber geometry contract as default selection.
    return matches[0]


def run(settings):
    import numpy as np
    import torch
    from .live_evaluate import freeze_checkpoint, run as trace
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
    torch.set_num_threads(1)
    torch.manual_seed(20260913)
    output = Path(settings["output"])
    previous = None
    if settings.get("resume", False):
        previous = json.loads((output / "report.json").read_text())
        from lasagna.inference_provenance import sha256_file
        snapshot, digest = Path(previous["checkpoint"]), previous["checkpoint_sha256"]
        if sha256_file(snapshot) != digest:
            raise ValueError("frozen pilot checkpoint changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot, digest = freeze_checkpoint(Path(settings["checkpoint"]), output)
    payload = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
    template = json.loads(Path(settings["prepared_config"]).read_text())
    source = next(s for s in template["sources"] if s["id"] == settings["source_id"])
    def length(path):
        fibers = load_fiber_file(Path(path))
        if len(fibers) != 1: raise ValueError("pilot requires single-fiber files")
        points = np.asarray(fibers[0].line_points_xyz)
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    fiber = select_pilot_fiber(source, settings.get("fiber"), length)
    source = {**source, "fibers": [fiber]}
    report = dict(checkpoint=str(snapshot), checkpoint_sha256=digest,
        step=payload["step"], samples_seen=payload.get("samples_seen"),
        fiber=fiber, length_base_voxels=length(fiber), settings=settings,
        note="One whole held-out fiber, both directions. Fresh predictions per variant; source caches are not purged. Non-exclusive GPU timings include co-running training.", variants=[])
    if previous is not None:
        if previous["fiber"] != fiber or previous["settings"]["prepared_config"] != settings["prepared_config"]:
            raise ValueError("cannot resume changed pilot selection")
        report = previous
    (output / "report.json").write_text(json.dumps(report, indent=2))
    for weight in settings.get("weights", [0, .25]):
        if any(v["weight"] == weight for v in report["variants"]):
            continue
        config = copy.deepcopy(template)
        tag = f"weight-{weight:g}".replace(".", "p")
        if (output / tag).exists():
            tag += f"-retry-{time.time_ns()}"
        config.update(sources=[source], output=str(output / tag), workers=2,
            decoded_cache_bytes=2 * 1024**3, render=False,
            inference_cache_namespace=f"embedding-pilot-{time.time_ns()}")
        config["inference"].update(tile_size=256, border=32, overlap=96,
            accumulator_dtype="float32", input_workers=4, input_io_threads=4,
            input_decode_threads=2, input_cache_bytes=2 * 1024**3, input_slots=8,
            result_slots=4, max_batch_size=1, slots_per_gpu=1,
            request_timeout_seconds=300, emit_embeddings=weight > 0)
        config["trace_args"] += ["--embedding-weight", str(weight),
            "--embedding-history-spacing-base-voxels", "32",
            "--embedding-history-window-base-voxels", "256", "--embedding-history-count", "8"]
        path = output / f"{tag}.json"
        path.write_text(json.dumps(config, indent=2))
        start = time.monotonic()
        print(f"embedding pilot {tag} step={report['step']} samples={report['samples_seen']} fiber={fiber}", flush=True)
        trace(path, snapshot, "0", settings["native_binary"], unmanaged=True)
        summary = json.loads((output / tag / "summary.json").read_text())
        report["variants"].append(dict(weight=weight, output=str(output / tag), seconds=time.monotonic()-start, summary=summary))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(f"embedding pilot {tag} finished seconds={report['variants'][-1]['seconds']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("--unmanaged", action="store_true")
    parser.add_argument("--devices", default="0")
    args = parser.parse_args()
    settings = json.loads(Path(args.control).read_text())["embedding_trace_pilot"]
    if not args.unmanaged:
        from .managed_evaluation import launch_managed
        return launch_managed([sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_pilot", args.control],
            devices=settings.get("devices", "0"), settings=settings["gpu_manager"])
    run(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
