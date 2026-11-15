#!/usr/bin/env python3
"""Move each fitted winding of a Spiral checkpoint toward the papyrus ridge in the CT.

    python ridge_correct_checkpoint.py checkpoint_fitted.ckpt corrected.ckpt --ct CT_ZARR_LEVEL

A post-fit step that uses no labels: for every node of the checkpoint's gap
grid it reads raw-CT intensity profiles across the fitted winding, estimates
how far the winding sits from the bright papyrus ridge, and writes that shift
into the gap parameters. The output is the same checkpoint with only
``gap_expander_params.logits`` changed (and ``model_state_sha256`` recomputed),
so the exporter and the flattener read it unchanged. It never changes a fit
unless it is run. See README.md, "CT-ridge correction of a fitted checkpoint".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ridge_correction import stage
from ridge_correction.ctstore import ZarrChunkStore


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Correct a fitted Spiral checkpoint toward the CT papyrus ridge.")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path, help="corrected checkpoint (never overwritten)")
    p.add_argument("--ct", required=True,
                   help="Zarr v2 uint8 CT level in the fit's frame (shape = the checkpoint's base_shape_zyx): "
                        "a local directory or an http(s) URL; read chunk by chunk, never copied")
    p.add_argument("--umbilicus", type=Path, help="umbilicus.json (auto-detected when omitted)")
    p.add_argument("--z-range", type=float, nargs=2, metavar=("Z0", "Z1"),
                   help="z band to correct (default: the checkpoint's z_begin..z_end)")
    p.add_argument("--windings", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   help="windings to correct (default: the windings the fit reports)")
    p.add_argument("--threads", type=int, default=1, help="worker processes (default: 1)")
    p.add_argument("--method", choices=stage.METHODS, default="tracking",
                   help="tracking (default): follow each ridge from confident nodes to agreeing neighbours; "
                        "nearest: move every confident node to its nearest ridge")
    p.add_argument("--field-scale", type=float, default=1.0,
                   help="fraction of the estimated shift to write (default 1.0; see README)")
    return p.parse_args(argv)


def run(args, log=stage.log_line, slab_rows=4, stop_after_slabs=None):
    """The command. ``slab_rows`` and ``stop_after_slabs`` exist for the tests."""
    t0 = time.time()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    report_path = Path(str(output) + ".json")
    work_dir = Path(str(output) + ".work")

    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    threadpool_limits(1)
    log(f"[ridge] loading {args.checkpoint}")
    fit = stage.load_fit(args.checkpoint, args.umbilicus)
    if stage.PROVENANCE_KEY in fit.checkpoint:
        raise ValueError("this checkpoint was already corrected by this tool; start from the original fit")
    ct = ZarrChunkStore(args.ct)
    base = fit.checkpoint.get("base_shape_zyx")
    if base is not None and tuple(int(v) for v in base) != ct.shape:
        raise ValueError(f"CT shape {ct.shape} differs from the checkpoint's base_shape_zyx {tuple(base)}: "
                         "pass the CT level the fit was made in")
    band = tuple(args.z_range) if args.z_range else (float(fit.checkpoint["z_begin"]), float(fit.checkpoint["z_end"]))
    windings = tuple(args.windings) if args.windings else stage.default_windings(fit)
    log(f"[ridge] CT {ct.location} shape {ct.shape}; band z {band[0]:g}-{band[1]:g}; work dir {work_dir} (resumes)")

    t_est = time.time()
    e = stage.estimate(fit, ct, band, windings=windings, work_dir=work_dir, threads=args.threads,
                       slab_rows=slab_rows, log=log, stop_after_slabs=stop_after_slabs, method=args.method)
    t_est = time.time() - t_est
    U, field_stats = stage.field(fit, e, band=band, windings=windings, rows=e["plan"].rows, scale=args.field_scale)
    report = {
        "tool": "ridge_correct_checkpoint",
        "input": str(fit.path), "output": str(output),
        "ct": {"location": ct.location, "shape": ct.shape, "chunks": ct.chunks, **ct.stats},
        "band": list(band), "windings": list(windings), "rows": list(e["plan"].rows), "slabs": len(e["plan"].slabs),
        "settings": stage.SETTINGS, "fidelity_bar": stage.FIDELITY,
        "memory_estimate_bytes": e["plan"].memory,
        "extraction": {"nan_profiles": int(sum(s["nan_profiles"] for s in e["slabs"])),
                       "on_demand_chunks": int(sum(s["on_demand_chunks"] for s in e["slabs"])),
                       "work_cpu_s": float(sum(s["work_cpu_s"] for s in e["slabs"])),
                       "slabs_resumed": int(sum(bool(s.get("resumed")) for s in e["slabs"]))},
        "field": field_stats,
        "template_centres": {str(k): v for k, v in e["centres"].items()},
    }
    edge = [k for k, v in e["centres"].items() if v and v["on_window_edge"]]
    if edge:
        log(f"[ridge] warning: template centre on the edge of its +-{stage.est.CENTRE_WINDOW:g} winding window "
            f"(ridge peak outside it) for windings {', '.join(map(str, edge))}")
    t_write = time.time()
    try:
        rep_write, Ueff = stage.write(fit, U, output, band=band, windings=windings,
                                      extra={"band": list(band), "windings": list(windings),
                                             "method": args.method, "field_scale": args.field_scale})
    except stage.FidelityError as err:
        report["write"] = getattr(err, "report", None)
        report["refused"] = str(err)
        _save_json(report_path, report)
        raise
    report["write"] = rep_write
    report["timing_s"] = {"estimate": t_est, "write_and_check": time.time() - t_write, "total": time.time() - t0}
    np.save(str(output) + ".field.npy", Ueff.astype(np.float32))
    _save_json(report_path, report)
    log(f"[ridge] wrote {output} ({rep_write['logits_changed']} logits changed); report {report_path}")
    return report


def _save_json(path, obj):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def main(argv=None) -> int:
    try:
        run(_parse_args(argv))
    except stage.FidelityError as err:
        print(f"[ridge] REFUSED: {err}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
