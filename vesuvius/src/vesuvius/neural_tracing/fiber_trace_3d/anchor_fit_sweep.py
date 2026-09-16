"""Reproducible placement experiment; all fitting remains in native C++."""
from copy import deepcopy
import json
from pathlib import Path
from lasagna.inference_provenance import atomic_write, sha256_file


def variants():
    yield 'legacy', None
    for iterations in (0, 2, 4, 8):
        for gamma in (1, 2):
            for q in ((0,) if iterations == 0 else (0, 2, 4)):
                yield f'weighted_i{iterations}_p{gamma}_q{q}', dict(
                    method='weighted', iterations=iterations, presence_exponent=gamma,
                    direction_exponent=q, local_sigma=1.5, damping=.5)


def run_sweep(workload, devices, gpu_manager=None):
    from .short_benchmark_evaluate import run
    root = Path(workload['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(workload=workload, binary_sha256=sha256_file(workload['native_binary']),
                    manifest_sha256=sha256_file(workload['benchmark_manifest']))
    provenance = root / 'sweep_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != identity:
            raise ValueError('sweep identity changed; use a new output root')
    else:
        atomic_write(provenance, identity)
    rows = []
    for name, fit in variants():
        current = deepcopy(workload)
        current.pop('anchor_sweep')
        current['output_root'] = str(root / name)
        current['quiet_report'] = True
        current['log_construction_chunks'] = False
        if fit is not None:
            current['fiberlet']['anchor_fit'] = fit
        metrics = Path(current['output_root']) / 'metrics.json'
        print(f'anchor-sweep start {name}', flush=True)
        if metrics.exists():
            saved = json.loads((metrics.parent / 'provenance.json').read_text())
            if saved['workload'] != current or saved['native_sha256'] != identity['binary_sha256']:
                raise ValueError('variant provenance mismatch')
            report = json.loads(metrics.read_text())
        else:
            report = run(current, devices, gpu_manager=gpu_manager)
        row = dict(name=name, fit=fit, output=current['output_root'],
                   wall_seconds=report['wall_seconds'], overall=report['overall'], scrolls=report['scrolls'])
        rows.append(row)
        atomic_write(root / 'matrix.json', rows)
        (root / 'matrix.md').write_text(matrix_markdown(rows))
        print(f"anchor-sweep done {name}: anchors={row['overall']['accepted_anchors']} "
              f"empty_bins={row['overall']['empty_arc_bins_percent']:.3f}%", flush=True)
    atomic_write(root / 'completed.json', dict(completed=True, variants=len(rows)))
    return rows


def matrix_markdown(rows):
    out = ["# Single-anchor weighted-fit experiment", "",
           "Same short200 GT, model,32-base-voxel cells and40/80ellipse. "
           "Legacy is existing multi-component/refined/NMS extraction; experimental rows change "
           "fitting, multiplicity and NMS together. Gamma=presence exponent, q=direction exponent.", "",
           "| Variant | Accepted anchors | Empty arc bins % | Gap P95 cells | Max gap cells | "
           "End max cells | Angle median ° | Angle P95 ° | ≤20° % | Empty bins requiring ≤20° % | Wall s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        s = row['overall']
        out.append(f"| {row['name']} | {s['accepted_anchors']} | {s['empty_arc_bins_percent']:.3f} | "
                   f"{s['interior_gap_cells']['p95']:.3f} | {s['interior_gap_cells']['maximum']:.3f} | "
                   f"{s['endpoint_gap_cells']['maximum']:.3f} | {s['direction']['median']:.2f} | "
                   f"{s['direction']['p95']:.2f} | {s['direction']['within_degrees_percent']['20']:.2f} | "
                   f"{s['direction_aware_empty_arc_bins_percent']['20']:.3f} | "
                   f"{row['wall_seconds']:.2f} |")
    out += ["", "Timings include setup and cache effects; not controlled throughput comparisons. "
            "Full per-scroll statistics and histograms: matrix.json and each variant's report/metrics. "
            "No downstream graph tracing in this placement-only sweep.", ""]
    return "\n".join(out)
