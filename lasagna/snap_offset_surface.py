"""Snap a Lasagna Offset surface onto the papyrus, check it, recover missed vertices (command line for offset_ct_snap).

	lasagna-offset-ct-snap --start START.tifxyz --fit FIT.tifxyz --offset 1 --ct VOLUME.zarr/0 --out OUT.tifxyz

--start                 tifxyz of the surface the Offset fit started from (winding N)
--fit                   tifxyz written by the Offset fit (same grid as --start, meant for winding N+1 or N-1)
--offset                the offset that was passed to the fit: +1 or -1
--ct                    the CT array (local path or http(s) / s3 URL of a Zarr array, uint8), read with vc_reader
--out                   output tifxyz folder; it must not exist
--smooth-within-paper   optional: smooth normal shifts within the CT papyrus layer to reduce >8-voxel neighbour jumps
--fallback-march        optional: when the air-gap check does not accept the surface, deliver it on other evidence:
                        the sheet the fit itself sits on when the two windings touch without air (delivery_basis
                        "fit_sheet"), or a march to the first sheet beyond the air when the fit went past the middle
                        of that air gap without landing (delivery_basis "ct_march", state 4). Refused otherwise.
--repair-start-pockets  optional: on accepted surfaces, bring back vertices left in a start-winding pocket (state 3 or
                        4). Transactional: a repair that does not bring a depression back to the level of the accepted
                        surface is rolled back and counted in start_pockets_rolled_back.
--auto-orient           optional: flip --offset sign when the fit moved strongly to the opposite side of the start grid

The output holds the dense snapped grid (x.tif, y.tif, z.tif; absent vertices are (-1, -1, -1)), the extra channel
snap_state.tif (0 not delivered, 1 anchored, 2 lifted, 3 recovered, 4 marched), meta.json and snap_report.json.
snap_report.json states the basis of the delivery (delivery_basis: air_gap / fit_sheet / ct_march / null), the measured
evidence it rests on (evidence) and, when nothing is delivered, the reason (refusal_reason).

Exit code: 0 surface accepted (one air gap crossed); 4 surface NOT accepted by the air-gap check but delivered by
--fallback-march on the stated basis (check it); 2 surface refused (the folder is written, nothing is delivered);
3 bad input or CT not available (nothing is written). See docs/offset_ct_snap.md.

The CT that was read is named in snap_report.json (location, shape, chunks, dtype, sha256 of the crop). When a surface
states the volume it lives in (base_shape_zyx in its meta.json) and --ct has another shape, the run stops: coordinates
of one volume must not be snapped onto the grey levels of another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import offset_ct_snap as ocs

BLOCK = 128


class InputError(Exception):
	"""Something is wrong with what was passed in; nothing has been written."""


def read_surface(path) -> tuple[np.ndarray, dict]:
	from vc_reader import tifxyz
	p = Path(path)
	if not p.is_dir():
		raise InputError(f"{p} is not a tifxyz folder")
	try:
		surf = tifxyz.read(p, mask=False)
	except Exception as err:
		raise InputError(f"{p}: cannot be read as tifxyz ({err})") from err
	if not np.asarray(surf.valid, dtype=bool).all():
		raise InputError(f"{p}: the grid has absent vertices; pass a window where every vertex is present")
	meta_file = p / "meta.json"
	meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
	return np.asarray(surf.xyz, dtype=np.float64), meta


def ct_crop(location: str, lo, hi, expect_shapes=()) -> tuple[np.ndarray, np.ndarray, dict]:
	"""CT block covering [lo, hi] (xyz), extended to 128-voxel boundaries: (array indexed z, y, x; xyz origin; identity).

	expect_shapes: (name, base_shape_zyx) pairs taken from the surfaces; the array must have each of these shapes."""
	from vc_reader.reader import open_zarr
	try:
		src = open_zarr(location)
	except Exception as err:
		raise InputError(f"cannot open the CT array {location} ({err})") from err
	shape = np.array([int(v) for v in src.shape])
	if len(shape) != 3:
		raise InputError(f"the CT array must be 3D (z, y, x), got shape {tuple(shape)}")
	for name, want in expect_shapes:
		if [int(v) for v in want] != [int(v) for v in shape]:
			raise InputError(f"{name} was made in a volume of shape {[int(v) for v in want]} (base_shape_zyx) but --ct has shape {[int(v) for v in shape]}: not the same volume")
	if np.dtype(src.dtype) != np.uint8:
		raise InputError(f"the CT array is {src.dtype}; the thresholds of this tool are 8-bit grey levels")
	lo = np.maximum(np.floor(np.asarray(lo) / BLOCK).astype(int) * BLOCK, 0)
	hi = np.minimum((np.floor(np.asarray(hi) / BLOCK).astype(int) + 1) * BLOCK, shape[::-1])
	if (hi <= lo).any():
		raise InputError(f"the surface lies outside the CT array of shape {tuple(shape)} (z, y, x)")
	try:
		vol = np.ascontiguousarray(src[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]])
	except Exception as err:
		raise InputError(f"cannot read CT block from {location} ({err})") from err
	identity = {
		"location": str(location), "shape_zyx": [int(v) for v in shape], "chunks_zyx": [int(v) for v in src.chunks], "dtype": str(np.dtype(src.dtype)),
		"crop_sha256": hashlib.sha256(vol.tobytes()).hexdigest(),
		"checked_against": [name for name, _ in expect_shapes] or "no surface states base_shape_zyx: shape not checked",
	}
	return vol, lo.astype(float), identity


def write_surface(out_dir: Path, xyz: np.ndarray, state: np.ndarray, meta: dict) -> None:
	"""Same layout as fit2tifxyz: meta.json, x/y/z as float32, extra channels declared in meta["extra_channels"]."""
	from vc_reader import images
	valid = state > 0
	out = np.where(valid[..., None], xyz, -1.0).astype(np.float32)
	if valid.any():
		pts = out[valid]
		meta["bbox"] = [[float(v) for v in pts.min(0)], [float(v) for v in pts.max(0)]]
	else:
		meta["bbox"] = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]          # nothing delivered: the empty box of fit2tifxyz
	max_state = int(max(3, state.max())) if state.size else 3
	meta["extra_channels"] = [{"name": "snap_state", "file": "snap_state.tif", "dtype": "uint8", "range": [0, max_state]}]
	try:
		out_dir.mkdir(parents=True, exist_ok=False)
	except FileExistsError as err:
		raise InputError(f"{out_dir} already exists; an existing surface is never overwritten") from err
	(out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	for k, name in enumerate(("x.tif", "y.tif", "z.tif")):
		images.write_tiff(str(out_dir / name), np.ascontiguousarray(out[..., k]), compression=None)
	images.write_tiff(str(out_dir / "snap_state.tif"), np.ascontiguousarray(state.astype(np.uint8)), compression=None)


def run(args) -> int:
	out_dir = Path(args.out)
	if out_dir.exists():
		raise InputError(f"{out_dir} already exists; an existing surface is never overwritten")
	if args.offset not in (1.0, -1.0):
		raise InputError("--offset must be +1 or -1")
	start, start_meta = read_surface(args.start)
	fit, meta = read_surface(args.fit)
	pts = np.concatenate([start.reshape(-1, 3), fit.reshape(-1, 3)])
	if args.roi:
		lo, hi = np.array(args.roi[:3]), np.array(args.roi[3:])
		if (hi <= lo).any():
			raise InputError("--roi must be X0 Y0 Z0 X1 Y1 Z1 with X1 > X0, Y1 > Y0 and Z1 > Z0")
		if (pts.max(0) < lo).any() or (pts.min(0) > hi).any():
			raise InputError("--roi does not touch the surfaces: nothing could be snapped")
	else:
		margin = ocs.PARAMS["march_max_vox"] + ocs.PARAMS["reach_vox"]
		lo, hi = pts.min(0) - margin, pts.max(0) + margin
	t0 = time.time()
	expect = [(name, m["base_shape_zyx"]) for name, m in (("--start", start_meta), ("--fit", meta)) if m.get("base_shape_zyx")]     # fit2tifxyz: shape of the volume the written coordinates are in, at any export scale
	vol, org, identity = ct_crop(args.ct, lo, hi, expect)
	res = ocs.snap_offset_surface(
		start,
		fit,
		vol,
		org,
		args.offset,
		smooth_within_paper=bool(getattr(args, "smooth_within_paper", False)),
		allow_march_fallback=bool(getattr(args, "fallback_march", False)),
		auto_orient=bool(getattr(args, "auto_orient", False)),
		repair_start_pockets=bool(getattr(args, "repair_start_pockets", False)),
	)
	dom = res["domain"]
	if not dom.any():
		raise InputError("no vertex has its whole profile inside the CT that was loaded: enlarge --roi, or check that the surfaces are in the voxel frame of --ct")
	state = np.where(dom, res["state"], 0)
	report = {
		"tool": "offset_ct_snap", "version": ocs.__version__,
		"verdict": res["verdict"], "accepted": res["accepted"],
		"vertices_in_domain": int(dom.sum()), "delivered": int((state > 0).sum()),
		"anchored": int((state == 1).sum()), "lifted": int((state == 2).sum()), "recovered": int((state == 3).sum()),
		"not_delivered": int((dom & (state == 0)).sum()),
		"grid_step_in": res["grid_step"], "grid_step_out": res["grid_step"] / max(1, int(round(res["grid_step"] / ocs.PARAMS["dense_step_vox"]))),
		"ct": str(args.ct), "ct_identity": identity, "ct_crop_origin_xyz": [int(v) for v in org], "ct_crop_shape_zyx": list(vol.shape),
		"start": str(args.start), "fit": str(args.fit), "offset": args.offset,
		"sources": {"start_uuid": start_meta.get("uuid"), "fit_uuid": meta.get("uuid")},
		"seconds": round(time.time() - t0, 1), "params": ocs.PARAMS,
	}
	report["marched"] = int((state == 4).sum())
	report["fallback_marched"] = bool(res.get("fallback_marched"))
	if res.get("fallback_marched") and res.get("fallback_height_from"):
		report["fallback_height_from"] = res["fallback_height_from"]
	if res.get("fallback_refused"):
		report["fallback_refused"] = res["fallback_refused"]
	report["median_fit_normal_move_vox"] = res.get("median_fit_normal_move_vox")
	report["valid_rays"] = res.get("valid_rays"); report["start_vertices_on_paper"] = res.get("start_vertices_on_paper")
	if res.get("fallback_target_height_vox") is not None:
		report["fallback_target_height_vox"] = res["fallback_target_height_vox"]
	report["delivery_basis"] = res.get("delivery_basis")
	report["refusal_reason"] = res.get("refusal_reason")
	report["evidence"] = res.get("evidence")
	if bool(getattr(args, "repair_start_pockets", False)):
		report["start_pockets_repaired"] = int(res.get("start_pockets_repaired", 0))
		report["start_pockets_rolled_back"] = int(res.get("start_pockets_rolled_back", 0))
	if res.get("auto_oriented"):
		report["auto_oriented"] = True
	if bool(getattr(args, "smooth_within_paper", False)):
		report["smoothed_within_paper"] = True
	keep = {k: v for k, v in meta.items() if k in ("type", "format", "base_shape_zyx", "lasagna_base_shape_zyx", "export_coordinate_scale")}
	new_meta = {
		"uuid": str(uuid.uuid4()), "name": out_dir.name, "type": "seg", "format": "tifxyz", **keep,
		"scale": [1.0 / report["grid_step_out"], 1.0 / report["grid_step_out"]],
		"offset_ct_snap": {k: v for k, v in report.items() if k != "params"},
	}
	write_surface(out_dir, res["xyz"], state, new_meta)
	(out_dir / "snap_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	print(json.dumps({k: v for k, v in report.items() if k != "params"}))
	if res["accepted"]:
		if res.get("start_pockets_repaired") or res.get("start_pockets_rolled_back"):
			print(f"offset_ct_snap: accepted; --repair-start-pockets kept {res.get('start_pockets_repaired', 0)} repaired vertices "
			      f"(snap_state 3 or 4) and rolled back {res.get('start_pockets_rolled_back', 0)} that did not bring a depression back to "
			      "the level of the accepted surface. Without the option the surface is delivered as the check accepted it.", file=sys.stderr)
		return 0
	if res.get("fallback_marched"):
		ev = res.get("evidence") or {}
		if res.get("delivery_basis") == "fit_sheet":
			how = (f"{report['delivered']} vertices delivered on the sheet the fit itself sits on (delivery_basis fit_sheet): "
			       f"{100 * (ev.get('no_gap_share') or 0):.0f} % of the rays cross no air gap, the windings touch; the step is "
			       f"{ev.get('step_height_vox')} voxels. No air gap confirms this step: which winding it is rests on the Offset fit alone")
		else:
			src = {"fit": "heights projected from the fit", "ct_consensus": "heights from the CT dark-gap consensus, not from the fit"}.get(res.get("fallback_height_from"), "unknown heights")
			how = (f"{report['marched']} vertices delivered by a march along the start normal (delivery_basis ct_march, snap_state 4), {src}; "
			       f"{res.get('valid_rays')} valid rays for {res.get('start_vertices_on_paper')} start vertices on paper. The march goes to the "
			       "first sheet beyond the air and does not check which winding that is")
		print(f"offset_ct_snap: NOT accepted (verdict {res['verdict']}); --fallback-march: {how}"
			f"{'; --offset was flipped by --auto-orient' if res.get('auto_oriented') else ''}. Check the surface before using it.", file=sys.stderr)
		return 4
	why = refusal_reason(res['verdict'], res.get('median_fit_normal_move_vox', 0.0))
	if res.get("fallback_refused"):
		why += f" --fallback-march delivered nothing: {res['fallback_refused']}."
	print(f"offset_ct_snap: refused, verdict {res['verdict']}: {why}", file=sys.stderr)
	return 2


def refusal_reason(verdict: int, median_move_vox: float = 0.0) -> str:
	if verdict == 0:
		if median_move_vox <= -ocs.PARAMS["min_forward_vox"]:
			return (f"the fit did not cross a dark gap along the requested offset direction (median move {median_move_vox:+.1f} voxels, "
				"towards the opposite side): check the sign of --offset or whether the start tifxyz grid is transposed (pass --auto-orient "
				"to orient from the fit). If most surfaces are refused with near-zero move, check lasagna/configs/offset.json vs offset_snap.json "
				"(PR #2418). See docs/offset_ct_snap.md, Requirements.")
		return ("the fit did not cross a dark gap from the start winding. If most surfaces are refused this way, the Offset fit "
			"probably did not reach the next winding: lasagna/configs/offset.json does not reach it on PHerc0139, and offset_snap.json "
			"must parse and set pred_dt to 0.0 in its snap stage (PR #2418). See docs/offset_ct_snap.md, Requirements.")
	if verdict == 2:
		return (f"the fit crossed two dark gaps, or fewer than {ocs.PARAMS['min_delivered_fraction']:.0%} of the vertices "
			"in the CT that was loaded could be placed on paper.")
	return "too few rays from the start winding could be read in the CT that was loaded (enlarge --roi?)."


def main(argv=None) -> int:
	ap = argparse.ArgumentParser(description="Snap a Lasagna Offset surface onto the papyrus, check it, recover missed vertices.")
	ap.add_argument("--start", required=True, help="tifxyz of the start surface (winding N)")
	ap.add_argument("--fit", required=True, help="tifxyz produced by the Offset fit (same grid as --start)")
	ap.add_argument("--offset", required=True, type=float, help="offset passed to the fit: +1 or -1")
	ap.add_argument("--ct", required=True, help="CT Zarr array (local path or URL), uint8")
	ap.add_argument("--out", required=True, help="output tifxyz folder (must not exist)")
	ap.add_argument("--roi", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
		help="only load the CT inside this box; vertices whose profile leaves it are outside the domain")
	ap.add_argument("--smooth-within-paper", action="store_true",
		help="smooth normal displacements within the CT papyrus layer (clamped to <=4 voxels and CT > paper threshold)")
	ap.add_argument("--fallback-march", action="store_true",
		help="when the air-gap check does not accept the surface but the fit moved forward: deliver the sheet the fit sits on "
		"(touching windings), or march to the first sheet beyond the air if the fit went past the middle of that gap "
		"(exit code 4; delivery_basis in the report); never changes an accepted surface")
	ap.add_argument("--repair-start-pockets", action="store_true",
		help="on accepted surfaces, bring back vertices left in a start-winding pocket (snap_state 3 or 4); transactional, "
		"repairs that are not justified are rolled back")
	ap.add_argument("--auto-orient", action="store_true",
		help="flip --offset sign automatically if the fit moved strongly to the opposite side of the start grid")
	args = ap.parse_args(argv)
	try:
		return run(args)
	except (InputError, ValueError) as err:
		print(f"offset_ct_snap: {err}", file=sys.stderr)
		return 3


if __name__ == "__main__":
	raise SystemExit(main())
