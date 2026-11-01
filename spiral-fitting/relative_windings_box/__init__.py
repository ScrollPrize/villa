"""Relative windings in a box around a point, from a surface prediction and Lasagna grad_mag.

Run from spiral-fitting/:

    uv run python -m relative_windings_box --prediction PRED --grad-mag GM --umbilicus UMB.json --point X,Y,Z
        --out RUN [--base-affine identity|M.json] [--box 512] [--seeds 200] [--rng 0]
        [--split-rule v2|v1|none] [--json]

In order, stopping at the first refusal (exit 2, nothing computed):
  1. checks the inputs: a surface prediction and a grad_mag source must both open (their metadata is read), and
     an umbilicus must be given. It never computes a prediction or a grad_mag;
  2. grows chains of sheet crossings (chain rule version 1, digest 0e53e0ae5016) on the prediction, from seeds
     drawn in a cube of --box voxels centered on the point, counting outward from the umbilicus: RUN/chains.json;
  3. samples gm_x per step from the grad_mag (RUN/features.json), cuts every chain at the steps the split rule
     flags (v2, params digest cd273aed7673; v1, c6d2eecbebc2; none, no cut: flags.json marks the steps v2 would
     cut as places to check, and export.json reports the run's gm_x median, p90 and share above 1.0)
     (RUN/split.json) and writes RUN/relative_windings.json (relative-winding point collections,
     winding_is_absolute false), RUN/flags.json (places to check, the cut steps among them) and RUN/export.json;
  4. prints a summary (with --json, one JSON object on stdout).

Exit codes: 0 written and both point collection files pass the checks, 1 written but a check failed (for example
no piece of two crossings in the box), 2 refused or failed. Needs numpy and scipy.

The same code as windlabel-box in ChaseHendrick/Scrolls (docs/tools/windlabel-box.md).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

VERSION = "0.1.0"
EXIT_OK, EXIT_CHECK, EXIT_REFUSED = 0, 1, 2
CHAIN_VERSION = 1
NOTE = "Model output from a surface prediction and lasagna grad_mag; not a reading."


class Refused(Exception):
    pass


def _parts():
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        raise Refused(f"relative_windings_box needs numpy and scipy ({exc})") from exc
    from . import _chains, _split
    return _chains, _split


def parse_point(text):
    try:
        v = [float(x) for x in str(text).split(",")]
    except ValueError:
        v = []
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise Refused(f"the point {text!r} is not x,y,z in level-0 voxels")
    return v


def parse_positive(text, what, kind=float):
    try:
        v = kind(str(text).strip())
    except ValueError:
        raise Refused(f"{what} {text!r} is not a number") from None
    if not (v > 0 and math.isfinite(v)):
        raise Refused(f"{what} must be above 0, not {text!r}")
    return v


def base_affine(text):
    """None for identity (or blank), else the matrix from a JSON file."""
    t = str(text or "").strip()
    if t in ("", "identity"):
        return None
    p = Path(t)
    if not p.is_file():
        raise Refused(f"--base-affine {t!r} is neither 'identity' nor a JSON matrix file")
    m = json.loads(p.read_text())
    m = m.get("matrix") if isinstance(m, dict) else m
    import numpy as np
    A = np.asarray(m, dtype=float)
    if A.shape not in ((3, 4), (4, 4)):
        raise Refused(f"--base-affine {t}: a 3 x 4 or 4 x 4 matrix on (x, y, z, 1), not {list(A.shape)}")
    return A.tolist()


def check_inputs(a):
    """The refusals that come before any work: (point, box edge, seeds, base affine)."""
    if not str(a.prediction or "").strip():
        raise Refused("no surface prediction given: name the volume's published surface prediction (for "
                      "example its m7 zarr). This tool does not compute one")
    if not str(a.grad_mag or "").strip():
        raise Refused("no grad_mag source given: name the scroll's published lasagna grad_mag (its manifest or its "
                      "grad_mag zarr). This tool does not compute one")
    if not str(a.umbilicus or "").strip():
        raise Refused("no umbilicus given: pass --umbilicus, the scroll's umbilicus JSON in the prediction's "
                      "level-0 voxels. Guessing the axis from the box's own sheets pointed outward the wrong way "
                      "for most chains in 4 of 6 PHerc0139 boxes of 512 voxels")
    if not Path(a.umbilicus).is_file():
        raise Refused(f"--umbilicus {a.umbilicus}: no such file")
    point = parse_point(a.point)
    edge = parse_positive(a.box, "--box")
    seeds = parse_positive(a.seeds, "--seeds", int)
    return point, edge, seeds, base_affine(a.base_affine)


def open_prediction(location):
    """Read the prediction's metadata, so a wrong path refuses before chains start."""
    from . import _zarr
    try:
        vol = _zarr.Volume(str(location), cache_chunks=8, workers=1)
        meta = vol.levels[0]["meta"]
    except Exception as exc:  # noqa: BLE001 - any failure to open is a refusal naming the source
        raise Refused(f"cannot open the surface prediction {location}: {type(exc).__name__}: {exc}") from exc
    return {"location": str(location), "shape": list(meta["shape"]), "chunks": list(meta["chunks"])}


def grow(wl, prediction, point, edge, seeds, rng, umbilicus, out):
    """Chains on a binary prediction around the point (chain rule version 1), written to `out`."""
    rule = dict(wl.RULES[CHAIN_VERSION])
    t0 = time.time()
    c0 = time.process_time()
    pred = wl._open_pred(prediction, 0, None)
    box = wl._box_at(",".join(repr(v) for v in point), edge / 2.0)
    seed_pts = wl.seeds_on(pred, box, seeds, rng)
    print(f"relative_windings_box: {len(seed_pts)} seeds", file=sys.stderr, flush=True)
    axis = wl.load_umbilicus(umbilicus)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    chains = wl.grow_chains(pred, seed_pts, axis, rule=rule, log=log)
    rec = {"rule": rule, "rule_digest": wl.rule_digest(rule), "prediction": prediction, "box": box,
           "seeds": seeds, "rng": rng, "axis": axis.tolist(), "axis_source": "file",
           "summary": wl.summary(chains), "seconds": round(time.time() - t0, 1),
           "cpu_seconds": round(time.process_time() - c0, 1), "chains": chains}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec))
    print(json.dumps(rec["summary"], indent=2), file=sys.stderr)
    return rec


def run(a):
    wl, ws = _parts()
    point, edge, seeds, base = check_inputs(a)
    pred = open_prediction(a.prediction)
    try:
        ws.open_grad_mag(a.grad_mag, base, a.gm_scaledown, a.gm_decode)
    except ws.SplitError as exc:
        raise Refused(str(exc)) from exc
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chains_path = out / "chains.json"
    try:
        grow(wl, str(a.prediction), point, edge, seeds, a.rng, a.umbilicus, chains_path)
    except wl.WindError as exc:
        raise Refused(str(exc)) from exc
    d = json.loads(chains_path.read_text())
    try:
        fd = ws.features_doc(chains_path, d, a.grad_mag, base, a.gm_scaledown, a.gm_decode, workers=a.workers)
    except ws.SplitError as exc:
        raise Refused(str(exc)) from exc
    (out / "features.json").write_text(json.dumps(fd))
    summ = ws.export_split(d, fd["steps"], out, params=dict(ws.SPLIT_RULES[a.split_rule]), chains_file=chains_path,
                           features_file=out / "features.json",
                           extra={"grad_mag": fd["grad_mag"], "features_seconds": fd["seconds"],
                                  "gm_x": ws.gm_stats(fd["steps"])})
    files = dict(summ["files"], **{"chains.json": str(chains_path), "features.json": str(out / "features.json")})
    summ["files"] = files
    summ["run"] = {"tool": f"relative_windings_box {VERSION}", "point": point, "box_edge": edge, "seeds": seeds,
                   "rng": a.rng, "prediction": pred, "chain_rule_version": CHAIN_VERSION,
                   "umbilicus": a.umbilicus, "wall_seconds": round(time.time() - t0, 2)}
    (out / "export.json").write_text(json.dumps(summ, indent=2))
    res = {"ok": summ["ok"], "files": files, "collections": summ["collections"], "points": summ["points"],
           "pieces": summ["pieces"], "steps": summ["steps"], "split_steps": summ["split_rule"]["flagged_steps"],
           "flags": summ["flags"], "checks": summ["checks"], "run": summ["run"], "gm_x": summ["gm_x"],
           "note": NOTE}
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for k, v in res.items():
            print(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return EXIT_OK if summ["ok"] else EXIT_CHECK


def build_parser(p=None):
    if p is None:
        p = argparse.ArgumentParser(prog="python -m relative_windings_box",
                                    description="Relative windings in a box around a point, from a surface "
                                                "prediction and Lasagna grad_mag.")
    p.add_argument("--prediction", default="", help="the volume's surface prediction: zarr folder or https URL "
                                                    "(binary, level 0, sheet above 127)")
    p.add_argument("--grad-mag", default="", help="lasagna grad_mag: a manifest (.json or *.lasagna.json URL) or a "
                                                  "grad_mag OME-Zarr")
    p.add_argument("--base-affine", default="identity",
                   help="'identity' or a JSON 3 x 4 matrix file from prediction voxels to the lasagna base grid")
    p.add_argument("--gm-scaledown", type=int, default=None,
                   help="grad_mag zarr level over the base (default: the manifest's, else 4)")
    p.add_argument("--gm-decode", type=float, default=None, help="grad_mag stored value to density per base voxel "
                                                                 "(default: the manifest's, else 0.25 / 1000)")
    p.add_argument("--point", required=True, help="the box center x,y,z in level-0 voxels")
    p.add_argument("--box", default="512", help="box edge in voxels (default 512)")
    p.add_argument("--seeds", default="200", help="seed count (default 200)")
    p.add_argument("--rng", type=int, default=0, help="seed RNG (default 0)")
    p.add_argument("--split-rule", default="v2", choices=("v2", "v1", "none"),
                   help="split rule: v2 cuts at gm_x above 1.0, v1 at 0.95, none keeps every step (default v2; "
                        "with none, flags.json marks the steps the v2 cut would remove as places to check)")
    p.add_argument("--umbilicus", help="umbilicus JSON in the prediction's level-0 voxels; it sets which way is "
                                       "outward")
    p.add_argument("--workers", type=int, default=2, help="threads for grad_mag reads (default 2)")
    p.add_argument("--out", required=True, help="run folder")
    p.add_argument("--json", action="store_true", help="print one JSON object")
    return p


def run_command(args):
    try:
        return run(args)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def main(argv=None):
    return run_command(build_parser().parse_args(argv))
