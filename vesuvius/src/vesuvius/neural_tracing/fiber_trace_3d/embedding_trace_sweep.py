"""Cached native history sweep, with optional manager-gated coverage extension."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import sys

from lasagna.inference_provenance import atomic_write, json_digest
from .cached_trace_profile import PublishedStore, extend_cached_replay
from .live_evaluate import coordinate_native, implementation_identity


def variants(controls_only=False, weight=.25):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("sweep weight must be finite and nonnegative")
    yield dict(name="off", weight=0, spacing=32, window=256, count=8, falloff="uniform")
    yield dict(name="original", weight=.25, spacing=32, window=256, count=8, falloff="uniform")
    if controls_only:
        return
    for spacing in (8, 16, 32):
        for window in (32, 64, 128, 256):
            for falloff in ("uniform", "linear", "gaussian"):
                yield dict(name=f"s{spacing}-w{window}-{falloff}", weight=weight,
                           spacing=spacing, window=window, count=64, falloff=falloff)


def replay_hash(output):
    return json_digest({p.parent.name: json.loads(p.read_text())["replay"]
                       for p in sorted((Path(output) / "cases").glob("*/trace.json"))})


def search_weight(execute, *, history=None, start_weight=.25):
    """Strict neighbor descent on 0.05 ticks; ties retain the current weight."""
    history = dict(history or dict(spacing=8, window=256, count=64, falloff="gaussian"))
    if not math.isfinite(start_weight) or start_weight < 0 or abs(start_weight * 20 - round(start_weight * 20)) > 1e-8:
        raise ValueError("start_weight must be a nonnegative multiple of .05")
    scores = {}
    def evaluate(tick):
        if tick not in scores:
            row = execute(dict(history, name=f"weight-{tick * 5:03d}", weight=tick / 20))
            if not row["completed"]:
                raise RuntimeError("incomplete weight replay cannot establish a local optimum")
            scores[tick] = row
        return scores[tick]
    current = round(start_weight * 20)
    while True:
        center = evaluate(current)
        neighbors = [t for t in (current - 1, current + 1) if t >= 0]
        for tick in neighbors:
            evaluate(tick)
        best = min([current] + neighbors, key=lambda t: scores[t]["failures"])
        if scores[best]["failures"] >= center["failures"]:
            return center
        current = best


def configured_variants(settings):
    custom = settings.get("candidates")
    if custom is None:
        return list(variants(settings.get("controls_only", False) or settings.get("weight_search", False),
                             settings.get("grid_weight", .25)))
    if settings.get("weight_search", False) or not isinstance(custom, list) or not custom:
        raise ValueError("candidates must be nonempty and cannot accompany weight_search")
    rows = list(variants(True))
    for i, candidate in enumerate(custom):
        if set(candidate) != {"weight", "spacing", "window", "count", "falloff"}:
            raise ValueError("candidate must contain only embedding weight/spacing/window/count/falloff")
        if any(not isinstance(candidate[k], (int, float)) or not math.isfinite(candidate[k]) or candidate[k] <= 0
               for k in ("spacing", "window")):
            raise ValueError("history distances must be finite and positive")
        if type(candidate['count']) is not int or not 1 <= candidate['count'] <= 64:
            raise ValueError("history count must be integer in [1,64]")
        if not isinstance(candidate['weight'], (int, float)) or not math.isfinite(candidate['weight']) or candidate['weight'] < 0:
            raise ValueError("weight must be finite and nonnegative")
        if candidate['falloff'] not in ('uniform', 'linear', 'gaussian'):
            raise ValueError("unsupported falloff")
        rows.append(dict(candidate, name=f"candidate-{i:03d}"))
    return rows


def run(settings):
    scheduled = configured_variants(settings)
    root = Path(settings["pilot"])
    pilot = json.loads((root / "report.json").read_text())
    previous = {"off" if v["weight"] == 0 else "original": Path(v["output"]) for v in pilot["variants"]}
    summary = json.loads((previous["original"] / "summary.json").read_text())
    if not summary["completed"]:
        raise ValueError("completed inference required")
    base = summary["config"]
    fill_missing = settings.get("fill_missing", False)
    for source in base["sources"]:
        identity = json.loads((Path(source["fiber_manifest"]).parent / "cache_identity.json").read_text())
        if identity["identity"]["model"]["checkpoint_sha256"] != pilot["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint mismatch")
        if fill_missing and identity["identity"]["model"]["implementation"] != implementation_identity():
            raise ValueError("inference implementation changed: cannot extend frozen cache")
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint_sha256=pilot["checkpoint_sha256"], step=pilot["step"],
                  fiber=pilot["fiber"], ranking="completed only, minimum bidirectional failures",
                  inference="fill missing only under manager" if fill_missing else "disabled; missing coverage fails variant", variants=[])

    def execute(variant):
        # Off used a separate (non-embedding) cache in the pilot. Reproduce it
        # there; all positive-weight comparisons use the same signed cache.
        evaluation = deepcopy(json.loads((previous["off"] / "summary.json").read_text())["config"]
                              if variant["weight"] == 0 else base)
        evaluation.pop("live_endpoint", None)
        evaluation.update(output=str(output / variant["name"]), workers=2, render=False)
        args = evaluation["trace_args"]
        # Remove only existing embedding option/value pairs, preserving all other settings.
        cleaned, i = [], 0
        while i < len(args):
            if args[i].startswith("--embedding-"):
                i += 2
            else:
                cleaned.append(args[i]); i += 1
        for flag, key in [("weight", "weight"), ("history-spacing-base-voxels", "spacing"),
                          ("history-window-base-voxels", "window"), ("history-count", "count"),
                          ("history-falloff", "falloff")]:
            cleaned += ["--embedding-" + flag, str(variant[key])]
        evaluation["trace_args"] = cleaned
        atomic_write(output / (variant["name"] + ".json"), evaluation)
        stores, mapping, channels = {}, {}, {}
        for source in evaluation["sources"]:
            key = source["fiber_manifest"]
            stores[key] = PublishedStore(key)
            mapping[source["id"]] = key
            channels[key] = stores[key].channels
        row = dict(variant)
        started = time.monotonic()
        print("sweep start " + variant["name"], flush=True)
        try:
            if fill_missing and variant["name"] not in previous and variant["name"] != "repeat-best":
                extend_cached_replay(evaluation, settings["native_binary"], Path(pilot["checkpoint"]))
            else:
                coordinate_native(evaluation, Path(settings["native_binary"]), stores, mapping, channels,
                    {k: set() for k in stores}, {}, (),
                    {"cached_trace_timeout_seconds": 180}, allow_inference=False)
            inventory = json.loads((Path(evaluation["output"]) / "inference_summary.json").read_text())
            if set(inventory["cache_roots"].values()) != {str(s.root) for s in stores.values()}:
                raise RuntimeError("replay did not reuse expected cache")
            row["inferred_tiles"] = sum(s["inferred_tiles"] for s in inventory["sources"].values())
            result = json.loads((Path(evaluation["output"]) / "summary.json").read_text())
            if not result["completed"] or result["cases"] != 2:
                raise RuntimeError("incomplete bidirectional replay")
            row.update(completed=True, failures=result["failures"], summary=result,
                       replay_hash=replay_hash(evaluation["output"]))
            if variant["name"] in previous:
                row["original_replay_match"] = row["replay_hash"] == replay_hash(previous[variant["name"]])
        except RuntimeError as error:
            if "cached-only inference miss" not in str(error):
                raise
            row.update(completed=False, error=str(error))
        row["seconds"] = time.monotonic() - started
        report["variants"].append(row)
        atomic_write(output / "report.json", report)
        print("sweep result " + json.dumps({k:v for k,v in row.items() if k != "summary"}), flush=True)
        return row

    for variant in scheduled:
        row = execute(variant)
        if variant["name"] in previous and (not row["completed"] or not row["original_replay_match"]):
            raise RuntimeError("control did not reproduce original replay; inspect before interpreting sweep")
    if settings.get("weight_search", False):
        best = search_weight(execute, history=settings.get('search_history'), start_weight=settings.get('start_weight', .25))
        best = min([best] + [v for v in report['variants'] if v['name'] in previous], key=lambda v: v['failures'])
        report["ranking"] = "local minimum of bidirectional failures on 0.05 weight grid; ties retain center"
    else:
        completed = [v for v in report["variants"] if v["completed"]]
        if len(completed) != len(scheduled):
            raise RuntimeError("full history grid incomplete; cannot select winner")
        best = min(completed, key=lambda v: v["failures"])
    repeated = execute({**{k:best[k] for k in ("weight", "spacing", "window", "count", "falloff")}, "name":"repeat-best"})
    report["best"] = best["name"]
    report["repeat_identical"] = repeated.get("replay_hash") == best["replay_hash"]
    atomic_write(output / "report.json", report)
    if not report["repeat_identical"]:
        raise RuntimeError("best replay is not deterministic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("--unmanaged", action="store_true")
    parser.add_argument("--devices", default="0")
    args = parser.parse_args()
    settings = json.loads(Path(args.control).read_text())["embedding_trace_sweep"]
    if settings.get("fill_missing", False) and not args.unmanaged:
        from .managed_evaluation import launch_managed
        return launch_managed([sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep", args.control],
                              devices="0", settings=settings["gpu_manager"])
    run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
