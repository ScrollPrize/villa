"""Summarize completed native AFV benchmarks; no extraction or input mutations."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def summarize(afv, directory):
    with sqlite3.connect(afv.resolve().as_uri() + "?mode=ro", uri=True) as db:
        count, arc = db.execute(
            "SELECT count(*),sum(length) FROM fibers"
        ).fetchone()
    rows = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get("runs") or data.get("normal_alignment") is not False:
            continue
        runs = data["runs"]
        if any(r["signed"] != 0 or r["unsigned_events"] != r["events"] for r in runs):
            raise ValueError(f"Unsigned benchmark contains unexpected signed events: {path}")
        times = [r["seconds"] for r in runs]
        counter_keys = [k for k in runs[0] if not k.endswith("seconds")]
        stable = all(all(r[k] == runs[0][k] for k in counter_keys) for r in runs)
        if not stable:
            raise ValueError(f"Repeated extraction counts differ: {path}")
        phases = {key: statistics.mean(r[key] for r in runs) for key in
                  ("prepare_seconds", "search_seconds", "orientation_seconds", "winding_seconds")}
        resource_name = "pilot256" if path.stem == "side256" else path.stem
        resource_file = directory / "runner" / (resource_name + ".resources.txt")
        resources = {}
        if resource_file.exists():
            for word in resource_file.read_text().split():
                if "=" in word:
                    key, value = word.split("=", 1)
                    resources[key] = float(value)
        rows.append(dict(
            name=path.stem, selected_fibers=data["selected_fibers"], arc_base=data["arc_base"],
            query_seconds=data["query_seconds"], load_seconds=data["load_seconds"],
            initial_extraction_s=times[0], warm_mean_s=statistics.mean(times[1:]) if len(times)>1 else None,
            extraction_mean_s=statistics.mean(times), extraction_median_s=statistics.median(times),
            extraction_min_s=min(times), extraction_max_s=max(times), repetitions=len(times),
            counts_repeat_identically=stable, counts={k: runs[0][k] for k in counter_keys},
            mean_phases=phases, resources=resources,
            conditional_full_arc_extraction_s=(statistics.mean(times) * arc / data["arc_base"]
                                               if runs[0]["events"] else None),
            conditional_full_arc_initial_s=times[0] * arc / data["arc_base"],
            conditional_full_arc_warm_s=(statistics.mean(times[1:]) * arc / data["arc_base"] if len(times)>1 else None),
        ))
    return dict(full_fibers=count, full_arc_base=arc, results=rows,
                caveat="Conditional linear extrapolations, not measured global runtimes. "
                "Subsets omit neighbors; candidate density, memory and I/O can change scaling. "
                "No normal alignment. First extraction has a fresh process cache; filesystem "
                "cache is uncontrolled. Later repetitions reuse process and filesystem caches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afv", type=Path, default=Path(__file__).parent / "data/Paris4_v1.afv")
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "data/unsigned_z_slab_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.afv, args.results), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
