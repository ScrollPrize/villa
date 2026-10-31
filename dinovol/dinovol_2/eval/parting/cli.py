"""Parting: at a contact between touching papyrus layers, score whether a candidate point lies on point a's layer.

    python -m dinovol_2.eval.parting sites --layer L0.vcps --layer L1.vcps [...] --shape Z Y X --out SITES.json
    python -m dinovol_2.eval.parting features --sites SITES.json --volume URL [--resample 3.24:7.91] --out RUN
    python -m dinovol_2.eval.parting score --run RUN --model MODEL.joblib --out READOUT.json [--sites SITES.json]
    python -m dinovol_2.eval.parting fit --run RUN [--run RUN2 ...] --out MODEL.joblib
    python -m dinovol_2.eval.parting compare RUN_A RUN_B [--model MODEL.joblib]
    python -m dinovol_2.eval.parting frag3 --out DIR (--sites-only | --model MODEL.joblib) [--device cuda]

A site is a point a on one layer with its normal, a second point a2 on the same layer and a point b on the next
layer at the same distance (sites). For each site, Parting reads the 128^3 CT window around it, runs the Vesuvius
Challenge's base dinovol encoder on it and keeps the tokens at the three points, CT profiles along each normal and
along a to a2 and a to b, and base dinovol's own token cosine (features). Its model, a soft vote of gradient
boosting and a one-layer MLP, scores the pair (a, a2) and the pair (a, b); a higher score means the same layer
(score). frag3 reruns the frozen Frag3 test from public data in one command. Model output only: Parting reads model
features of the scan; it does not read ink or text.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import classifier as M
from . import geometry

VERSION = "1.0.0"


def read_grid(path):
    """An (H, W, 3) xyz grid from a .vcps pointset, an .npy array or a tifxyz folder (x.tif, y.tif, z.tif)."""
    p = Path(path)
    if p.is_dir():
        import tifffile
        g = np.stack([tifffile.imread(p / f"{c}.tif") for c in "xyz"], -1).astype(np.float64)
        g[(g < 0).any(-1)] = -1
        return g
    if p.suffix == ".vcps":
        return geometry.read_vcps(p)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float64)
    raise SystemExit(f"{path}: give a .vcps pointset, an .npy grid or a tifxyz folder")


def load_sites(path):
    doc = json.loads(Path(path).read_text())
    if "sites" not in doc:
        raise SystemExit(f"{path}: no sites list")
    return doc


def cmd_sites(a):
    if len(a.layer) < 2:
        raise SystemExit("give at least two --layer grids, in stack order")
    grids = {}
    for n, path in enumerate(a.layer):
        g = read_grid(path)[::a.row_step, ::a.col_step] * a.scale
        grids[f"L{n}"] = g.astype(np.float32)
    report, cross = geometry.stack_layers(grids, a.need) if a.exclude_crossings else ({}, None)
    rule = geometry.Rule(seed=a.seed, dense=a.dense, spacing=a.spacing,
                         caps=dict(zip(("touching", "apart", "fused", "mixed"), a.caps)),
                         stop_when_full=a.stop_when_full, exclude=cross, exclude_vox=a.exclude_crossings or 0.0)
    sites = geometry.make_sites(grids, tuple(a.shape), rule)
    doc = {"kind": "parting-contact-sites", "layers": {f"L{n}": str(p) for n, p in enumerate(a.layer)},
           "layers_sha256": {f"L{n}": geometry.sha256_file(p) for n, p in enumerate(a.layer) if Path(p).is_file()},
           "scale": a.scale, "row_step": a.row_step, "col_step": a.col_step, "shape_zyx": list(a.shape),
           "stack": report, "rule": rule.record(), "sites_sha256": geometry.sites_sha256(sites), "sites": sites}
    Path(a.out).write_text(json.dumps(doc))
    print(json.dumps(geometry.tally(sites)))
    return 0


def pick(sites, ids=None, limit=None):
    if ids:
        want = set(ids)
        sites = [s for s in sites if s["id"] in want]
    if limit:
        sites = sites[:limit]
    return sites


def run_features(sites, volume, resample, out, device="cpu", checkpoint=None, threads=None, every=10):
    """Features for every site into the run folder out, resuming what is there; returns the run record."""
    from . import ct as CT
    from . import encoder
    from . import features as FE
    t0 = time.time()
    out = Path(out)
    feats, scores = FE.load_done(out)
    prev = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else None
    if prev and (prev.get("volume") != volume or prev.get("resample") != resample):
        raise SystemExit(f"{out} holds sites read from another volume or resampling; use a new --out")
    vol = CT.open_volume(volume, resample)
    enc = encoder.Encoder(device, checkpoint, threads=threads)
    todo = [s for s in sorted(sites, key=lambda s: tuple(s["centre"][::-1])) if s["id"] not in feats]
    print(f"{len(sites)} sites on {device}; {len(sites) - len(todo)} done before", flush=True)
    meta = {"kind": "parting-feature-run", "version": VERSION, "volume": volume, "resample": resample,
            "shape_zyx": list(vol.shape), "encoder": enc.record(), "groups": FE.GROUPS, "profile_vox": FE.PROFILE,
            "segment_samples": FE.SEG}
    for n, s in enumerate(todo, 1):
        ts = time.time()
        ct = vol.read(FE.window_lo(s))
        tr = time.time() - ts
        f, sc = FE.site_features(s, ct, enc.tokens(ct))
        feats[s["id"]], scores[s["id"]] = f, sc
        print(f"PROGRESS {n}/{len(todo)} site {s['id']} {s['stratum']} read {tr:.1f}s total {time.time() - ts:.1f}s",
              flush=True)
        if n % every == 0:
            FE.save_run(out, sites, feats, scores, meta)
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["sites"] = len(feats)
    FE.save_run(out, sites, feats, scores, meta)
    return meta


def cmd_features(a):
    doc = load_sites(a.sites)
    sites = pick(doc["sites"], a.ids, a.limit)
    meta = run_features(sites, a.volume, a.resample, a.out, a.device, a.checkpoint, a.threads)
    print(json.dumps({"run": a.out, "sites": meta["sites"], "wall_s": meta["wall_s"]}))
    return 0


def score_run(run, model_path, trust=False, sites=None, judge=True, boot=M.BOOT, permutations=None):
    mdl, rec = M.load_model(model_path, trust)
    rows, F = M.load_run(run)
    X, y, grp, strat, ctrl = M.design(rows, F)
    s = mdl.predict_proba(X)[:, 1]
    doc = M.readout(rows, s, y, grp, strat, ctrl, boot=boot, permutations=permutations, judge=judge, sites=sites)
    doc.update(kind="parting-readout", run=str(run), model=rec,
               features_sha256=M.sha256(Path(run) / "features.npz"))
    return doc, (rows, s, y, grp, strat, ctrl)


def cmd_score(a):
    sites = load_sites(a.sites)["sites"] if a.sites else None
    doc, _ = score_run(a.run, a.model, a.trust_model, sites, a.verdict, a.boot, a.permutations)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("primary", "permutation_gate", "verdict")}, indent=1))
    return 0


def cmd_fit(a):
    print(json.dumps(M.fit(a.run, a.out)))
    return 0


def compare_runs(ra, rb, model_path=None, trust=False):
    """Largest absolute differences between two feature runs on the sites both hold, per group and score, and for
    Parting's own score when a model is given."""
    rows_a, Fa = M.load_run(ra)
    rows_b, Fb = M.load_run(rb)
    ia = {r["id"]: n for n, r in enumerate(rows_a)}
    ib = {r["id"]: n for n, r in enumerate(rows_b)}
    both = sorted(set(ia) & set(ib))
    if not both:
        raise SystemExit(f"{ra} and {rb} share no site")
    na, nb = [ia[i] for i in both], [ib[i] for i in both]
    out = {"sites": len(both), "ids": both, "groups": {}, "scores": {}}
    for g in Fa:
        d = np.abs(Fa[g][na] - Fb[g][nb])
        out["groups"][g] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
    for k in M.SCALARS:
        va = np.array([rows_a[n]["scores"][k] for n in na], float)
        vb = np.array([rows_b[n]["scores"][k] for n in nb], float)
        out["scores"][k] = {"max_abs": float(np.abs(va - vb).max())}
    if model_path:
        mdl, rec = M.load_model(model_path, trust)
        pa = mdl.predict_proba(M.design([rows_a[n] for n in na], {g: v[na] for g, v in Fa.items()})[0])[:, 1]
        pb = mdl.predict_proba(M.design([rows_b[n] for n in nb], {g: v[nb] for g, v in Fb.items()})[0])[:, 1]
        out["parting"] = {"max_abs": float(np.abs(pa - pb).max()), "mean_abs": float(np.abs(pa - pb).mean()),
                          "model_sha256": rec["sha256"]}
        same = np.sign(pa[0::2] - pa[1::2]) == np.sign(pb[0::2] - pb[1::2])
        out["parting"]["same_order_sites"] = f"{int(same.sum())}/{len(same)}"
    return out


def cmd_compare(a):
    print(json.dumps(compare_runs(a.run_a, a.run_b, a.model, a.trust_model), indent=1))
    return 0


def frag3_sites(out, pointsets=None, download=True):
    """The frozen Frag3 site list, rebuilt from the public pointsets (or read back from OUT/frag3_sites.json, which a
    first run writes) and checked against the frozen hash: (sites, record)."""
    from . import frag3 as F3
    cached = Path(out) / "frag3_sites.json"
    if cached.exists():
        doc = load_sites(cached)
        if geometry.sites_sha256(doc["sites"]) == F3.SITES_SHA:
            return doc["sites"], {"from": str(cached), "sites_sha256": F3.SITES_SHA}
    folder = Path(pointsets or Path(out) / "pointsets")
    if download:
        F3.fetch_pointsets(folder)
    sites, rec = F3.rebuild_sites(folder)
    if rec["sites_sha256"] != F3.SITES_SHA:
        raise SystemExit(f"sites made from the pointsets (sha256 {rec['sites_sha256']}) differ from the frozen "
                         f"site list ({F3.SITES_SHA})")
    doc = {"kind": "parting-contact-sites", "object": "PHercParis1Fr34 (Frag3)", "volume": F3.VOLUME,
           "resample": F3.RESAMPLE, "shape_zyx": list(F3.SHAPE_ZYX),
           "layers": rec["layers"],
           **{k: rec[k] for k in ("stack", "crossings", "rule", "sites_sha256")}, "sites": sites}
    cached.write_text(json.dumps(doc))
    print(f"sites rebuilt from the pointsets match the frozen list ({len(sites)} sites)", flush=True)
    return sites, {"from": str(folder), **{k: rec[k] for k in ("layers", "stack", "crossings", "sites_sha256")}}


def cmd_frag3(a):
    from . import frag3 as F3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_sites, srec = frag3_sites(out, a.pointsets, not a.no_download)
    record = {"kind": "parting-frag3", "version": VERSION, "sites": srec, "tally": geometry.tally(all_sites)}
    if a.sites_only:
        (out / "frag3.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record, indent=1))
        return 0
    if not a.model:
        raise SystemExit("frag3: give --model (the released SharpMeow/parting-v1 file, or one from fit with --trust-model) or --sites-only")
    sites = pick(all_sites, a.ids, a.limit)
    full = len(sites) == len(all_sites)
    meta = run_features(sites, F3.VOLUME, F3.RESAMPLE, out / "run", a.device, a.checkpoint, a.threads)
    rd, (rows, s, y, grp, strat, ctrl) = score_run(out / "run", a.model, a.trust_model, all_sites, judge=full)
    top = F3.TOP_PAIR
    rd["reported_not_judged"] = {"without_" + top: M.without_pair(s, y, grp, strat, ctrl, top)}
    (out / "readout.json").write_text(json.dumps(rd, indent=1))
    exp = F3.EXPECTED
    if full:
        cmp = {"primary": {k: {"frozen": v, "here": rd["primary"][k], "diff": round(rd["primary"][k] - v, 4)}
                           for k, v in exp["primary"].items()},
               "verdict": {"frozen": exp["verdict"], "here": rd["verdict"]}}
    else:
        cmp = "subset: not judged; compare it per site with the compare command against a full run"
    record.update(features={"sites": meta["sites"], "wall_s": meta["wall_s"], "device": a.device},
                  readout="readout.json", judged=full, against_frozen=cmp)
    (out / "frag3.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"verdict": rd["verdict"], "primary": rd["primary"], "against_frozen": cmp}, indent=1))
    return 0


def parser():
    p = argparse.ArgumentParser(prog="python -m dinovol_2.eval.parting", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"parting {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sites", help="contact sites from layer grids (geometry only)")
    s.add_argument("--layer", action="append", required=True,
                   help="a .vcps, .npy or tifxyz grid; repeat in stack order")
    s.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"), help="volume shape, voxels")
    s.add_argument("--out", required=True)
    s.add_argument("--scale", type=float, default=1.0, help="multiply grid coordinates by this (voxel size change)")
    s.add_argument("--row-step", type=int, default=1, help="keep every Nth grid row")
    s.add_argument("--col-step", type=int, default=1, help="keep every Nth grid column")
    s.add_argument("--seed", type=int, default=geometry.SEED)
    s.add_argument("--dense", type=int, default=geometry.DENSE, help="sub-steps per grid cell")
    s.add_argument("--spacing", type=float, default=geometry.SPACING)
    s.add_argument("--caps", type=int, nargs=4, metavar=("TOUCHING", "APART", "FUSED", "MIXED"),
                   default=[geometry.CAPS[k] for k in ("touching", "apart", "fused", "mixed")])
    s.add_argument("--stop-when-full", action="store_true", help="leave a pair once every stratum is at its cap")
    s.add_argument("--exclude-crossings", type=float, default=0.0, metavar="VOX",
                   help="keep windows this far from vertices whose next layer is not above them (0: off)")
    s.add_argument("--need", type=float, default=0.995, help="with --exclude-crossings: share of matched vertices "
                   "whose next layer must lie above")
    f = sub.add_parser("features", help="base dinovol and CT features per site")
    f.add_argument("--sites", required=True)
    f.add_argument("--volume", required=True, help="a public zarr v2 uint8 array with 128^3 chunks (a level URL)")
    f.add_argument("--resample", help="SRC_UM:DST_UM: read a finer level box-averaged to DST_UM voxels")
    f.add_argument("--out", required=True, help="run folder (resumed when it exists)")
    f.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    f.add_argument("--checkpoint", help="a local copy of teacher_305k.pt (else downloaded from the Hub)")
    f.add_argument("--threads", type=int, help="torch CPU threads")
    f.add_argument("--ids", type=int, nargs="+", help="only these site ids")
    f.add_argument("--limit", type=int, help="only the first N sites")
    c = sub.add_parser("score", help="Parting's scores, the readout and, with --verdict, the gate")
    c.add_argument("--run", required=True)
    c.add_argument("--model", required=True, help="the joblib model file")
    c.add_argument("--out", required=True)
    c.add_argument("--sites", help="the site file, for the gap-at-most-4 row")
    c.add_argument("--verdict", action="store_true", help="apply the gate (PASS, FAIL or VOID)")
    c.add_argument("--trust-model", action="store_true", help="load a model file that is not a known one, such as one you made with fit")
    c.add_argument("--boot", type=int, default=M.BOOT, help="layer-pair resamples for the bounds")
    c.add_argument("--permutations", type=int, help="label permutations for the gate (default 1000)")
    t = sub.add_parser("fit", help="fit the recipe on feature runs (touching sites only)")
    t.add_argument("--run", action="append", required=True)
    t.add_argument("--out", required=True)
    m = sub.add_parser("compare", help="largest differences between two feature runs")
    m.add_argument("run_a")
    m.add_argument("run_b")
    m.add_argument("--model", help="also compare Parting's scores with this model")
    m.add_argument("--trust-model", action="store_true")
    g = sub.add_parser("frag3", help="rerun the frozen Frag3 test from public data")
    g.add_argument("--model", help="the released model (SharpMeow/parting-v1) or a file from fit (needed unless --sites-only)")
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    g.add_argument("--checkpoint", help="a local copy of teacher_305k.pt")
    g.add_argument("--threads", type=int)
    g.add_argument("--ids", type=int, nargs="+", help="only these site ids (not judged)")
    g.add_argument("--limit", type=int, help="only the first N sites (not judged)")
    g.add_argument("--pointsets", help="folder of SEGMENT/pointset.vcps (default OUT/pointsets)")
    g.add_argument("--no-download", action="store_true", help="use the pointsets already in the folder")
    g.add_argument("--sites-only", action="store_true", help="rebuild and check the sites, read no CT")
    g.add_argument("--trust-model", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    return {"sites": cmd_sites, "features": cmd_features, "score": cmd_score, "fit": cmd_fit,
            "compare": cmd_compare, "frag3": cmd_frag3}[a.cmd](a)

