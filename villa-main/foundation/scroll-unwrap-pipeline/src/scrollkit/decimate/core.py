"""M2 decimation engine.

Library policy (mesh-io-conventions, binding):
  * pymeshlab is an IN-MEMORY ENGINE ONLY. The source PLY is loaded into pymeshlab
    (VCG reference loader), cross-checked against scrollkit.io.read_ply (vertex/face
    counts identical, max-abs float32 vertex diff == 0, wedge texcoords present and
    bit-equal), the texture-aware quadric collapse runs, and the result is EXTRACTED
    as numpy arrays. pymeshlab never writes a file.
  * All disk IO goes through scrollkit.io (write_obj/copy_texture).

Filter used (pymeshlab 2025.7): meshing_decimation_quadric_edge_collapse_with_texture
with parameters targetperc / qualitythr / extratcoordw / preserveboundary /
boundaryweight / optimalplacement / preservenormal (names verified against the
installed wheel via pymeshlab.print_filter_parameter_list).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from scrollkit.io import MeshData, ObjFrameWriter, copy_texture, read_obj, read_ply, write_obj

from . import metrics

DECIMATION_FILTER = "meshing_decimation_quadric_edge_collapse_with_texture"
HAUSDORFF_SAMPLE_CAP = 500_000

# Group C (full-scroll photogrammetry) keep-ladder extension — policy
# Policy: decimate aggressively ONLY when visuals are unaffected.
# Those models are already coarse relative to their 4-8K textures: collapses move
# close-up texture by pixels, so the pinned ladder max (0.35) fails the close-up SSIM
# gate. C walks these extra conservative rungs before giving up on decimation.
GROUP_C_EXTENDED_LADDER = [0.50, 0.65, 0.80]

# Recorded verbatim (pinned) on every no_safe_decimation decision.
NO_SAFE_DECIMATION_RATIONALE = (
    "model is not oversampled relative to its texture; any reduction is perceptually "
    "lossy; quality dominates"
)


def ladder_for_group(cfg: dict, group: str) -> list[float]:
    """Keep-ladder for one mesh group: the pinned configs/global.yaml ladder, plus the
    GROUP_C_EXTENDED_LADDER rungs for Group C only (see constant note above)."""
    ladder = [float(x) for x in cfg.get("keep_ladder", [0.05, 0.08, 0.12, 0.20, 0.35])]
    if group == "C":
        ladder += [float(x) for x in GROUP_C_EXTENDED_LADDER]
    return ladder

DEFAULT_PARAMS = {
    "qualitythr": 0.5,
    "preserveboundary": True,
    "boundaryweight": 1.0,
    "extratcoordw": 1.0,
    "optimalplacement": True,
    "preservenormal": True,
    # Not pinned by configs/global.yaml; VCG quality option adding a planar-fitting
    # term to the quadric. Tried (recorded per attempt) for meshes failing perceptual
    # gates at the ladder top before escalating.
    "planarquadric": False,
}

UV_GATE_KEYS = ("uv_flips", "uv_charts")  # failures that justify the extratcoordw=3 retry


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class MeshDecimator:
    """Holds one source mesh (pymeshlab MeshSet id 0 + scrollkit arrays + gate stats)
    and decimates copies of it at requested keep-fractions."""

    def __init__(self, src_path: str | Path, edge_len_mean: float | None = None):
        import pymeshlab  # local import: workers initialize their own copy

        self.src_path = Path(src_path)
        self.mesh = read_ply(self.src_path)
        self.ms = pymeshlab.MeshSet()
        self.ms.load_new_mesh(str(self.src_path))
        pm = self.ms.current_mesh()
        self.notes: list[str] = []

        # ---- cross-check pymeshlab loader vs scrollkit.io (binding contract) ----
        if pm.vertex_number() != self.mesh.n_vertices:
            raise AssertionError(f"{self.src_path}: pymeshlab vertex count {pm.vertex_number()} != {self.mesh.n_vertices}")
        if pm.face_number() != self.mesh.n_faces:
            raise AssertionError(f"{self.src_path}: pymeshlab face count {pm.face_number()} != {self.mesh.n_faces}")
        vdiff = float(np.abs(pm.vertex_matrix().astype(np.float32) - self.mesh.vertices).max())
        if vdiff != 0.0:
            raise AssertionError(f"{self.src_path}: pymeshlab vertex coords differ (max abs {vdiff})")
        if not np.array_equal(pm.face_matrix().astype(np.int32), self.mesh.faces):
            raise AssertionError(f"{self.src_path}: pymeshlab face indices differ")
        if not pm.has_wedge_tex_coord():
            if pm.has_vertex_tex_coord():
                # only vertex texcoords made it through the loader: transfer so the
                # texture-aware collapse actually sees wedge UVs.
                self.ms.compute_texcoord_transfer_vertex_to_wedge()
                pm = self.ms.current_mesh()
                self.notes.append("vertex->wedge texcoord transfer applied (loader gave vertex UVs only)")
            else:
                raise AssertionError(f"{self.src_path}: no texcoords in pymeshlab mesh")
        assert pm.has_wedge_tex_coord(), "wedge texcoords required for the texture filter"
        if self.mesh.wedge_uv is not None:
            w = pm.wedge_tex_coord_matrix().reshape(-1, 3, 2).astype(np.float32)
            if not np.array_equal(w.view(np.uint32), self.mesh.wedge_uv.view(np.uint32)):
                raise AssertionError(f"{self.src_path}: pymeshlab wedge UVs differ from scrollkit reader")
        self.src_id = 0

        # ---- source-side gate baselines ----
        self.src_stats = metrics.mesh_stats(self.mesh.vertices, self.mesh.faces, self.mesh.wedge_uv)
        self.edge_len_mean = (
            float(edge_len_mean)
            if edge_len_mean is not None
            else metrics.mean_edge_length(self.mesh.vertices, self.mesh.faces)
        )
        self.last: dict | None = None  # arrays + stats of the most recent attempt

    # ------------------------------------------------------------------ decimation
    def _decimate_copy(self, keep: float, params: dict) -> int:
        """Copy source mesh, run the texture-aware quadric collapse, drop unreferenced
        vertices. Returns the new mesh id (left as current)."""
        self.ms.set_current_mesh(self.src_id)
        self.ms.generate_copy_of_current_mesh()
        self.ms.meshing_decimation_quadric_edge_collapse_with_texture(
            targetperc=float(keep),
            qualitythr=float(params["qualitythr"]),
            extratcoordw=float(params["extratcoordw"]),
            preserveboundary=bool(params["preserveboundary"]),
            boundaryweight=float(params["boundaryweight"]),
            optimalplacement=bool(params["optimalplacement"]),
            preservenormal=bool(params["preservenormal"]),
            planarquadric=bool(params.get("planarquadric", False)),
        )
        self.ms.meshing_remove_unreferenced_vertices()
        return self.ms.current_mesh_id()

    def _hausdorff_two_sided(self, dec_id: int) -> dict:
        nv_src = self.ms.mesh(self.src_id).vertex_number()
        nv_dec = self.ms.mesh(dec_id).vertex_number()
        fwd = self.ms.get_hausdorff_distance(
            sampledmesh=self.src_id, targetmesh=dec_id,
            samplenum=min(HAUSDORFF_SAMPLE_CAP, nv_src), samplevert=True, sampleface=True,
        )
        rev = self.ms.get_hausdorff_distance(
            sampledmesh=dec_id, targetmesh=self.src_id,
            samplenum=min(HAUSDORFF_SAMPLE_CAP, nv_dec), samplevert=True, sampleface=True,
        )
        return {
            "fwd": {k: float(fwd[k]) for k in ("max", "mean", "RMS")} | {"n_samples": int(fwd["n_samples"])},
            "rev": {k: float(rev[k]) for k in ("max", "mean", "RMS")} | {"n_samples": int(rev["n_samples"])},
            "two_sided_max": float(max(fwd["max"], rev["max"])),
            "two_sided_mean": float(max(fwd["mean"], rev["mean"])),
        }

    def try_rung(self, keep: float, extratcoordw: float, *, hausdorff_ratio_max: float = 1.0,
                 boundary_tol: float = 0.02, planarquadric: bool = False) -> dict:
        """Decimate from the ORIGINAL at `keep`, evaluate all geometry gates.

        Stores extracted arrays in self.last (whether passing or not) and removes the
        pymeshlab copy before returning. Returns the attempt record."""
        t0 = time.time()
        params = dict(DEFAULT_PARAMS)
        params["extratcoordw"] = float(extratcoordw)
        params["planarquadric"] = bool(planarquadric)
        dec_id = self._decimate_copy(keep, params)
        d = self.ms.current_mesh()

        V = np.ascontiguousarray(d.vertex_matrix()).astype(np.float32)
        F = np.ascontiguousarray(d.face_matrix()).astype(np.int32)
        W = np.ascontiguousarray(d.wedge_tex_coord_matrix()).reshape(-1, 3, 2).astype(np.float32)
        N = _normalize_rows(np.ascontiguousarray(d.vertex_normal_matrix()).astype(np.float64)).astype(np.float32)

        # The texture-aware collapse occasionally inverts a handful of tiny UV slivers
        # (a few texels, interior). Repair UVs only — geometry untouched — by local
        # Laplacian untangling of the offending UV-vertices; the flip gate below then
        # measures the repaired table honestly (and SSIM verifies perception later).
        uv_repair = None
        if metrics.uv_orientation_counts(W)["flipped"] > self.src_stats["uv_flipped"]:
            W, uv_repair = metrics.repair_uv_flips(F, W)

        hd = self._hausdorff_two_sided(dec_id)
        dec_stats = metrics.mesh_stats(V, F, W)

        fails: list[str] = []
        threshold = hausdorff_ratio_max * self.edge_len_mean
        if hd["two_sided_max"] > threshold:
            fails.append(f"hausdorff: two-sided max {hd['two_sided_max']:.3f} > {threshold:.3f} (= {hausdorff_ratio_max} x edge_len_mean)")
        if dec_stats["uv_flipped"] > self.src_stats["uv_flipped"]:
            fails.append(f"uv_flips: {dec_stats['uv_flipped']} > source {self.src_stats['uv_flipped']}")
        if dec_stats["uv_charts"] > self.src_stats["uv_charts"]:
            fails.append(f"uv_charts: {dec_stats['uv_charts']} > source {self.src_stats['uv_charts']}")
        src_bl = self.src_stats["boundary_length"]
        if src_bl == 0.0:
            if dec_stats["boundary_length"] != 0.0:
                fails.append(f"boundary: source closed but decimated has boundary length {dec_stats['boundary_length']:.3f}")
            bl_rel = 0.0
        else:
            bl_rel = abs(dec_stats["boundary_length"] - src_bl) / src_bl
            if bl_rel > boundary_tol:
                fails.append(f"boundary: length deviates {bl_rel:.4f} > {boundary_tol}")
        if dec_stats["nonmanifold_edge_count"] > self.src_stats["nonmanifold_edge_count"]:
            fails.append(f"nonmanifold: {dec_stats['nonmanifold_edge_count']} > source {self.src_stats['nonmanifold_edge_count']}")
        if dec_stats["isolated_vertices"] != 0:
            fails.append(f"isolated vertices: {dec_stats['isolated_vertices']}")

        # remove the pymeshlab copy (arrays already extracted)
        self.ms.set_current_mesh(dec_id)
        self.ms.delete_current_mesh()
        self.ms.set_current_mesh(self.src_id)

        gate_fail_kinds = sorted({f.split(":", 1)[0] for f in fails})
        attempt = {
            "keep": float(keep),
            "extratcoordw": float(extratcoordw),
            "planarquadric": bool(planarquadric),
            "faces_out": int(F.shape[0]),
            "verts_out": int(V.shape[0]),
            "keep_achieved": float(F.shape[0] / self.mesh.n_faces),
            "hausdorff": hd,
            "hausdorff_ratio": float(hd["two_sided_max"] / self.edge_len_mean) if self.edge_len_mean else None,
            "uv_repair": uv_repair,
            "dec_stats": dec_stats,
            "boundary_rel_diff": float(bl_rel),
            "pass": not fails,
            "fail_reasons": fails,
            "uv_only_failure": bool(fails) and all(k in UV_GATE_KEYS for k in gate_fail_kinds),
            "seconds": round(time.time() - t0, 2),
        }
        self.last = {"V": V, "F": F, "W": W, "N": N, "attempt": attempt}
        return attempt

    # ------------------------------------------------------------------ output
    def source_texture(self) -> str | None:
        """First declared texture that exists next to the source PLY (M1 logic)."""
        for cand in self.mesh.texture_files:
            if (self.src_path.parent / cand).exists():
                return cand
        if self.mesh.texture_files:
            raise FileNotFoundError(f"{self.src_path}: no declared texture exists on disk")
        return None

    def write_outputs(self, out_dir: str | Path, stem: str, group: str) -> dict:
        """Write <stem>_decimated.obj/.mtl + byte-copied texture from self.last arrays.

        A/B: per-vertex-UV (shared) OBJ when the wedge->vertex conversion is bit-exact;
        otherwise the wedge-dedup path (recorded). Group C always wedge-dedup."""
        assert self.last is not None, "try_rung must run before write_outputs"
        V, F, W, N = self.last["V"], self.last["F"], self.last["W"], self.last["N"]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tex_name = self.source_texture()
        tex_sha = copy_texture(self.src_path.parent / tex_name, out_dir / tex_name) if tex_name else None

        vertex_uv = None
        if group in ("A", "B"):
            vertex_uv = metrics.wedge_to_vertex_uv_exact(V.shape[0], F, W)
            uv_mode = "shared" if vertex_uv is not None else "wedge"
            if vertex_uv is None:
                self.notes.append("wedge UVs disagree at >=1 vertex after collapse - wedge-dedup OBJ path used")
        else:
            uv_mode = "wedge"
        normals = N if group in ("A", "B") else None  # C sources carry no normals (match M1)

        md = MeshData(
            vertices=V, faces=F, normals=normals,
            vertex_uv=vertex_uv, wedge_uv=W,
            texture_files=[tex_name] if tex_name else [],
            source_path=str(self.src_path),
        )
        obj_path = out_dir / f"{stem}_decimated.obj"
        write_obj(
            obj_path, md, texture_file=tex_name, uv_mode=uv_mode,
            header_comment=(
                f"M2 decimation of {self.src_path.name} | keep={self.last['attempt']['keep']} "
                f"| extratcoordw={self.last['attempt']['extratcoordw']} | uv_mode {uv_mode}"
            ),
        )
        frame_est = self._frame_size_estimate(md, uv_mode, tex_name)
        return {
            "obj": str(obj_path),
            "mtl": str(obj_path.with_suffix(".mtl")) if tex_name else None,
            "texture": tex_name,
            "texture_sha256": tex_sha,
            "uv_obj_mode": uv_mode,
            "obj_size_bytes": int(obj_path.stat().st_size),
            "frame_obj_est_bytes": frame_est,
        }

    def _frame_size_estimate(self, md: MeshData, uv_mode: str, tex_name: str | None) -> int:
        """Exact byte size of one animation-frame OBJ for this decimated mesh.

        Frames are written by ObjFrameWriter (no vn block). For the wedge path the
        frame writer is not applicable, so we measure a normal-less write_obj instead."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "frame.obj"
            if uv_mode == "shared" and md.vertex_uv is not None:
                fw = ObjFrameWriter(md, texture_file=tex_name, mtl_name="frame.mtl")
                fw.write_frame(tmp, md.vertices)
            else:
                md2 = MeshData(
                    vertices=md.vertices, faces=md.faces, normals=None,
                    vertex_uv=md.vertex_uv, wedge_uv=md.wedge_uv,
                    texture_files=md.texture_files, source_path=md.source_path,
                )
                write_obj(tmp, md2, texture_file=tex_name, uv_mode=uv_mode)
            return int(tmp.stat().st_size)


# ---------------------------------------------------------------------- ladder driver
def run_ladder(task: dict, cfg: dict) -> dict:
    """Full per-mesh M2 geometry pass.

    task: {group, stem, src (abs path), edge_len_mean, out_dir (abs path)}
    cfg:  decimation block of configs/global.yaml.

    Walks the group's keep ladder (ladder_for_group — Group C gets the extended rungs)
    from most aggressive up; a rung failing ONLY UV gates is retried once with
    extratcoordw=3.0 (UV-smearing remedy) before moving up. Outputs are written for the
    chosen rung (or the best-Hausdorff rung if none passed, with geometry_pass=false
    for the gate to flag)."""
    t0 = time.time()
    ladder = ladder_for_group(cfg, task["group"])
    ratio_max = float(cfg.get("hausdorff_max_mean_edge", 1.0))
    base_w = float(cfg.get("extratcoordw", 1.0))

    dec = MeshDecimator(task["src"], edge_len_mean=task["edge_len_mean"])
    journey: list[dict] = []
    chosen: dict | None = None
    for keep in ladder:
        att = dec.try_rung(keep, base_w, hausdorff_ratio_max=ratio_max)
        journey.append(att)
        if att["pass"]:
            chosen = att
            break
        if att["uv_only_failure"]:
            att3 = dec.try_rung(keep, 3.0, hausdorff_ratio_max=ratio_max)
            journey.append(att3)
            if att3["pass"]:
                chosen = att3
                break

    geometry_pass = chosen is not None
    if chosen is None:
        # nothing passed even at 0.35 — keep the best attempt for inspection, flag red.
        best = min(journey, key=lambda a: a["hausdorff"]["two_sided_max"])
        dec.try_rung(best["keep"], best["extratcoordw"], hausdorff_ratio_max=ratio_max)
        chosen = dec.last["attempt"]

    out = dec.write_outputs(task["out_dir"], task["stem"], task["group"])
    rec = {
        "group": task["group"],
        "stem": task["stem"],
        "src": task["src"],
        "faces_in": dec.mesh.n_faces,
        "verts_in": dec.mesh.n_vertices,
        "edge_len_mean": dec.edge_len_mean,
        "rung_chosen": chosen["keep"],
        "extratcoordw": chosen["extratcoordw"],
        "keep_achieved": chosen["keep_achieved"],
        "faces_out": chosen["faces_out"],
        "verts_out": chosen["verts_out"],
        "hausdorff": chosen["hausdorff"],
        "hausdorff_ratio": chosen["hausdorff_ratio"],
        "src_stats": dec.src_stats,
        "dec_stats": chosen["dec_stats"],
        "boundary_rel_diff": chosen["boundary_rel_diff"],
        "geometry_pass": geometry_pass,
        "fail_reasons": chosen["fail_reasons"],
        "journey": journey,
        "notes": dec.notes,
        "ssim": "pending",
        **out,
        "seconds_geometry": round(time.time() - t0, 2),
    }
    return rec


def redecimate_at(task: dict, cfg: dict, keep: float, extratcoordw: float,
                  planarquadric: bool = False) -> dict:
    """Re-run a single rung (used by the SSIM pass when a rung is bumped/retried),
    re-evaluating geometry gates. Outputs are rewritten ONLY when the geometry gates
    pass, so the on-disk OBJ always matches the last green record. Returns a partial
    record with out=None on geometry failure."""
    ratio_max = float(cfg.get("hausdorff_max_mean_edge", 1.0))
    dec = MeshDecimator(task["src"], edge_len_mean=task["edge_len_mean"])
    att = dec.try_rung(keep, extratcoordw, hausdorff_ratio_max=ratio_max,
                       planarquadric=planarquadric)
    out = dec.write_outputs(task["out_dir"], task["stem"], task["group"]) if att["pass"] else None
    return {
        "attempt": att,
        "out": out,
        "src_stats": dec.src_stats,
        "notes": dec.notes,
    }


def ship_undecimated_copy(task: dict, root: str | Path) -> dict:
    """No-safe-decimation fallback (Group C only): the 'optimized' deliverable is the
    UNDECIMATED M1 conversion — byte-identical copies of the M1 OBJ (renamed
    <stem>_decimated.obj), its MTL (original filename: the copied OBJ's mtllib line
    references it) and texture, each SHA256-verified. The copy is then re-read and
    proven bit-equal to the source PLY (V/F/wedge-UV), so the geometry gates hold by
    identity (two-sided Hausdorff = 0, stats unchanged).

    Returns a record fragment with decision='no_safe_decimation', the verbatim
    pinned rationale, copy provenance + SHAs, and write_outputs()-shaped keys."""
    root = Path(root)
    manifest = json.loads((root / "outputs/obj/manifest.json").read_text())
    src_resolved = Path(task["src"]).resolve()
    entry = next(
        (e for e in manifest["meshes"] if (root / e["src"]).resolve() == src_resolved),
        None,
    )
    if entry is None:
        raise FileNotFoundError(f"no M1 manifest entry for {task['src']}")

    m1_obj = root / entry["obj"]
    m1_mtl = root / entry["mtl"]
    tex_name = entry["texture"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_obj = out_dir / f"{task['stem']}_decimated.obj"
    dst_mtl = out_dir / m1_mtl.name

    def _copy_sha_verified(src: Path, dst: Path) -> str:
        shutil.copyfile(src, dst)
        h_src = hashlib.sha256(src.read_bytes()).hexdigest()
        h_dst = hashlib.sha256(dst.read_bytes()).hexdigest()
        if h_src != h_dst:
            raise IOError(f"copy corrupted: {src} -> {dst}")
        return h_src

    obj_sha = _copy_sha_verified(m1_obj, dst_obj)
    mtl_sha = _copy_sha_verified(m1_mtl, dst_mtl)
    tex_sha = copy_texture(m1_obj.parent / tex_name, out_dir / tex_name)
    if tex_sha != entry["texture_sha256"]:
        raise IOError(
            f"{task['stem']}: texture SHA {tex_sha} != M1 manifest {entry['texture_sha256']}"
        )
    stale_mtl = dst_obj.with_suffix(".mtl")
    if stale_mtl != dst_mtl and stale_mtl.exists():
        stale_mtl.unlink()  # MTL of the superseded decimated rung — copied OBJ references dst_mtl

    # Prove the deliverable IS the source mesh (bit-exact, mirroring the M1 gate).
    src_mesh = read_ply(task["src"])
    dec_mesh = read_obj(dst_obj)
    if dec_mesh.n_vertices != src_mesh.n_vertices or dec_mesh.n_faces != src_mesh.n_faces:
        raise AssertionError(f"{dst_obj}: copied V/F counts differ from source PLY")
    if not np.array_equal(dec_mesh.vertices.view(np.uint32), src_mesh.vertices.view(np.uint32)):
        raise AssertionError(f"{dst_obj}: copied vertices not bit-equal to source PLY")
    if not np.array_equal(dec_mesh.faces, src_mesh.faces):
        raise AssertionError(f"{dst_obj}: copied faces differ from source PLY")
    wedge = dec_mesh.wedge_uv
    if wedge is None and dec_mesh.vertex_uv is not None:
        wedge = dec_mesh.vertex_uv[dec_mesh.faces]
    if wedge is None or src_mesh.wedge_uv is None or not np.array_equal(
        wedge.view(np.uint32), src_mesh.wedge_uv.view(np.uint32)
    ):
        raise AssertionError(f"{dst_obj}: copied wedge UVs not bit-equal to source PLY")

    stats = metrics.mesh_stats(src_mesh.vertices, src_mesh.faces, src_mesh.wedge_uv)
    zero = {"max": 0.0, "mean": 0.0, "RMS": 0.0, "n_samples": 0}
    return {
        "decision": "no_safe_decimation",
        "decision_rationale": NO_SAFE_DECIMATION_RATIONALE,
        "deliverable": {
            "kind": "undecimated_m1_copy",
            "m1_obj": str(m1_obj),
            "m1_mtl": str(m1_mtl),
            "obj_sha256": obj_sha,
            "mtl_sha256": mtl_sha,
            "bit_equal_to_source_ply": True,
        },
        "rung_chosen": 1.0,
        "extratcoordw": None,
        "planarquadric": None,
        "keep_achieved": 1.0,
        "faces_out": src_mesh.n_faces,
        "verts_out": src_mesh.n_vertices,
        "hausdorff": {
            "fwd": dict(zero), "rev": dict(zero),
            "two_sided_max": 0.0, "two_sided_mean": 0.0,
            "basis": "identity: deliverable proven bit-equal to source PLY",
        },
        "hausdorff_ratio": 0.0,
        "src_stats": stats,
        "dec_stats": stats,
        "boundary_rel_diff": 0.0,
        "geometry_pass": True,
        "fail_reasons": [],
        "obj": str(dst_obj),
        "mtl": str(dst_mtl),
        "texture": tex_name,
        "texture_sha256": tex_sha,
        "uv_obj_mode": entry["uv_mode"],
        "obj_size_bytes": int(dst_obj.stat().st_size),
        # C meshes are never frame-exported (disk budget counts A/B wraps only); the
        # whole-OBJ byte size is the honest stand-in the gate requires to be recorded.
        "frame_obj_est_bytes": int(dst_obj.stat().st_size),
    }
