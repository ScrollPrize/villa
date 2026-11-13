"""High-level API: raw projection folders -> calibrated -> reconstructed xy slices.

    from helirec.recon import HelicalRecon
    r = HelicalRecon("/path/to/projections", cache_dir="/path/to/cache", device="cuda")
    slices = r.reconstruct_fractions([0.4, 0.5])          # (n, N, N) float32, attenuation in cm^-1

Slice positions use the reference conventions: a fraction f of the doable view-height span
[lo, hi) maps to view height h = hi - 1 - round_half_even(f * (hi - lo)) (f = 0 bottom, 1 top).

Modes (params["mode"]):
  corrected          (default) attenuation in cm^-1, periodic opposite-view rebalance, ramp padding reflected about
                     the true rotation axis, mending over the merged columns the scans cover, fixed unit-DC Paganin
                     kernel (when a Paganin filter is requested); no projection filters by default.
  legacy_reference   reproduces reconstructions of the reference workflow (Night Rail BM18 / nabu-helical / nabuxx):
                     its conventions (half-attenuation "reference units",
                     nabuxx rebalance endpoints, integer padding reflection, index-as-width mending bounds,
                     chunk-truncated Paganin) and every parameter pinned to LEGACY_REFERENCE. Pinned parameters cannot
                     be changed in this mode. Of the geometry calibration it takes the merged frame alone: one rotation
                     axis and one vector between the scans, with the recorded angles and heights.

The geometry of every projection is the geometry calibration's (geometry.py, helirec-calibration geometry: geometry.json, made
on first use and kept in the cache directory, or given: params["geometry"]); the darks, flats, weights and the
detector's blur field are calibration.py's.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import _core, bliss, calibration
from .sld import sld_config
from .filters import DEFAULT_CHAIN, REFERENCE_CHAIN, is_filter_key, override, parse_chain

MODES = ("corrected", "legacy_reference")

# default reconstruction parameters: the reference workflow's (Night Rail BM18 defaults) with the corrected conventions
# (mode "corrected"), no projection filters, and fbp (it takes each slice's angles: angle_interp, rolling_shutter)
DEFAULTS = dict(
    mode="corrected", filters=DEFAULT_CHAIN, margin=-1, n_grid=0, angular_tolerance_steps=5.0,
    luminosity_repartition=100.0, mend_threshold=1e-4, log_clip_min=1e-6, log_clip_max=10.0, block_angles=500,
    backprojector="fbp", grid_size=0, ghbp_legs=8, ghbp_reduction_steps=2, clip_outer_circle=True,
    axis_source_meters=1e9, sld="", x_interp="sinc", z_interp="lanczos8", angle_interp="views", exposure_arc=0.0,
    angle_subset="", sharpness_weighting="auto", sharp_f=0.35, sharp_band="0.15:0.30",
    geometry="", row_time_us="",
)
# geometry: "" (default: geometry.json in the cache directory, made there on first use by the geometry calibration
#   step: geometry.calibrate_geometry, helirec-calibration geometry; about 13 minutes on PHerc0343P) | the path of one made
#   before. It is the only geometry calibration: every projection is read on the grid of its detector map (rows by
#   the row kernel at the mapped row, columns resampled exactly) at the pose it gives the frame
#   (geometry.frame_poses: row, axis column, their slopes and scales), at the angle of its rows (the calibration's
#   row timing), in a merged frame laid out from the same poses (geometry.merged_layout). Needs x_interp=sinc; the
#   row timing needs angle_interp=views or nearest and fbp. provenance()["geometry"] says what was applied and what
#   the calibration measured and the reconstruction cannot express. Mode legacy_reference takes the merged frame
#   alone. The darks, flats, weights and the detector's blur field are calibration.py's (cached beside it).
# row_time_us: for the geometry calibration step, when it has to be made and no stretch of the scans holds a refill
#   of the ring to measure the rows' timing from: the time of a detector row within the frame, in microseconds per
#   stored row (signed; a property of the camera's configuration). "" (default): measured.
# sharpness_weighting (detector_smear until 2026-11-11, still accepted): where a ray is recorded more than once, its
#   recordings are combined above a crossover frequency with weights that favour the one recorded where the detector
#   is sharper. "auto" (default: the detector's blur field the calibration measured from the flats' noise,
#   Calibration.detector_blur_field; none if nothing significant) | "off" or ""
#   | "a11:a12:a21:a22:c1:c2": that blur field given outright, for the band-dependent redundancy weights
#   (pipeline.hpp); sharp_f, sharp_band = "f1:f2": the frequency the weights are taken at, and the crossover band.
# z_interp: lanczos8 (default since 2026-11-12) | lanczos3 | cubic | linear, the vertical interpolation of a slice's
#   (fractional) detector row. lanczos8 (16 rows) keeps the signal along the axis to 0.40 cycles per row (0.90 of its
#   amplitude at 0.45) where lanczos3 (6 rows, the default before) keeps 0.82 and 0.68; next to unmeasured rows and
#   the detector's ends both fall back to linear.
# angle_interp: views (default since 2026-11-11: every projection in a sinogram row of its own, filtered and
#   backprojected at its own angle; views of a slice a turn apart, or a turn at rest and the helix, share a grid angle
#   but were not recorded at the same angle, by the turn's excess over a whole number of steps and by the rows'
#   timing between them, and are no longer averaged at their mean angle. The sinogram has as many rows as bins the
#   chunk's projections span: 1.2 turns' worth in a helix of PHerc0343P's pitch, 2.2 where a turn at rest is among
#   them, and memory and time of the filters and the backprojection grow with it) | nearest (each projection whole
#   in one bin of one turn, slices backprojected at the bins' measured mean angles) | linear (the reference's two-bin
#   angular splat). views needs the exact backprojector (fbp, bands): the whole-slice methods use nearest, and ghbp,
#   whose plan fixes its angles, keeps the splat: info["angle_interp"] records the gridding in use. With views the
#   exposure arc is not deconvolved (exposure_arc is disabled for now; info["exposure_arc_bins"] is 0).
# exposure_arc: fraction of the angular step swept during an exposure, deconvolved along the angle; 0 (default since
#   2026-11-11) leaves the sinogram as recorded; "auto" takes it from the scan
#   (bliss.BlissScan.exposure_arc; 0 for step scans). Disabled for now with angle_interp=views (the default): the
#   filter runs along one turn of grid angles and is not carried over to a row per view; angle_interp=nearest has it.
# angle_subset: "" (default: every projection) or "I/N": subset I of N interleaved subsets of the projections, chosen
#   by angle (the same angles in every scan), reconstructed on a grid of 1 / N of the angles. The subsets have
#   independent noise and share the calibration: "0/2" and "1/2" are the two halves whose shell correlation
#   (evaluation/shell_correlation.py --pair) measures the scan's own signal-to-noise ratio.

# mode "legacy_reference": every parameter pinned to the reference workflow
LEGACY_REFERENCE = dict(
    filters=REFERENCE_CHAIN, margin=-1, n_grid=0, angular_tolerance_steps=5.0,
    luminosity_repartition=100.0, mend_threshold=1e-4, log_clip_min=1e-6, log_clip_max=10.0, block_angles=500,
    backprojector="ghbp", grid_size=0, ghbp_legs=8, ghbp_reduction_steps=2, clip_outer_circle=True,
    axis_source_meters=1e9, sld="", x_interp="linear", z_interp="linear", angle_interp="linear", exposure_arc=0.0,
    angle_subset="", sharpness_weighting="", sharp_f=0.35, sharp_band="0.15:0.30",
)
# what names the calibration rather than the reconstruction: not pinned by mode legacy_reference
CALIBRATION_KEYS = ("geometry", "row_time_us")
assert set(LEGACY_REFERENCE) == set(DEFAULTS) - {"mode", *CALIBRATION_KEYS}
# parameters of the calibration the geometry calibration replaced (2026-11-12): named in the error for older configs
REMOVED = ("z_translation_scale", "companion_registration", "stationary_registration", "axis_refinement", "rolling_shutter")

# uint8 display window of the reference workflow (legacy reference units); the same physical window in
# attenuation (cm^-1) is twice as wide
REFERENCE_WINDOW = (-0.03, 0.145)


def display_window(mode):
    """default uint8 window for outputs of `mode` (the reference window in that mode's units)"""
    k = 1.0 if mode == "legacy_reference" else 2.0
    return tuple(k * v for v in REFERENCE_WINDOW)


# conventions of each mode (provenance)
CONVENTIONS = {
    "corrected": dict(output_units="attenuation_cm-1", ramp_angular_weight="2pi/n_angles",
                      rebalance="periodic_opposite_view", ramp_padding_reflection="2*rot_center (interpolated)",
                      mending_domain="merged columns covered by the scans' weight maps",
                      paganin_kernel="fixed, rows [-m, m], unit DC",
                      column_offset="x_interp (sinc by default: rows shifted exactly; lanczos3 for measured-only gridding); "
                                    "linear splat next to unmeasured columns",
                      luminosity_window="full 25 rows (12 context rows read either side of a chunk)",
                      row_position="z_interp kernel (lanczos8 by default); linear next to unmeasured rows",
                      angular_gridding="angle_interp (views by default: a sinogram row per projection, each at its "
                                       "measured angle; nearest: one bin of one turn, the bin's mean measured angle; "
                                       "ghbp and linear: two-bin splat)",
                      exposure_arc="as recorded by default; exposure_arc=auto deconvolves the scan's recorded arc along the angle "
                                   "(disabled with angle_interp=views)"),
    "legacy_reference": dict(output_units="reference_units (attenuation / 2)", ramp_angular_weight="pi/n_angles",
                             rebalance="nabuxx (endpoint quirks)", ramp_padding_reflection="lround(2*rot_center)",
                             mending_domain="main-scan first/last column as left/right widths",
                             paganin_kernel="nabuxx pag_version 2 (truncated per chunk, DC not renormalised)",
                             column_offset="two-column linear splat", luminosity_window="clipped to the chunk",
                             row_position="linear", angular_gridding="two-bin splat", exposure_arc="none"),
}


def _chain_norm(chain):
    """filter chain with numeric parameter values as floats (1000 == 1000.0 == '1000')"""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return [{k: (v if k == "type" else num(v)) for k, v in f.items()} for f in parse_chain(chain)]


def _same(key, a, b):
    if key == "filters":
        return _chain_norm(a) == _chain_norm(b)
    if isinstance(b, bool) or isinstance(a, bool):
        return bool(a) == bool(b)
    if isinstance(b, (int, float)):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def legacy_reference_config():
    """SliceReconstructor config keys of mode legacy_reference (all but geometry / physics / device), for callers
    that build the reconstructor themselves"""
    p = {k: v for k, v in LEGACY_REFERENCE.items() if k not in ("backprojector", "sld")}
    p["filters"] = [dict(f) for f in LEGACY_REFERENCE["filters"]]
    return dict(p, legacy_reference=True, **backprojector_config(LEGACY_REFERENCE["backprojector"]))


def backprojector_config(spec):
    """Backprojector spec -> reconstructor config keys. Specs (text or dict):
      ghbp                                  the reference hierarchical backprojector
      fbp(interp=linear|cubic|lanczos3|bspline3|bspline5, oversample=1..8)   exact pixel-driven backprojector: the
                                            default (interp=bspline3, oversample=2: ~ideal interpolation)
                                            (oversample: FFT upsampling of the filtered rows before interpolating)
                                            ghbp and fbp also take the ramp filter, as gridrec:
                                            filter_name=ramlak (default)|shepp|cosine|hann|hamming|parzen|butterworth|none,
                                            filter_par=p0:p1
      sirt(num_iter=, relax=, norm=rowsum, weighting=mask|confidence, projector=sf|chord, init=constant|gridrec, init_value=)
      mlem(num_iter=, projector=sf|chord, init=constant|gridrec, init_value=)
      gridrec(filter_name=shepp|ramlak|..., filter_par=a:b)
                                            whole-slice methods (tomo.hpp)
      python(fn=NAME, ...)                  a function registered in helirec.backprojectors"""
    from .filters import parse_chain
    s = dict(spec) if isinstance(spec, dict) else parse_chain(str(spec))[0]
    typ = s.pop("type")
    if typ in ("ghbp", "fbp"):
        out = dict(backprojector=typ)
        if typ == "fbp":
            out.update(bp_interp=str(s.pop("interp", "bspline3")), bp_oversample=int(s.pop("oversample", 2)))
        for k in ("filter_name", "filter_par"):   # the ramp filter (C++ parses and checks it)
            if k in s:
                out["ramp_" + k] = str(s.pop(k))
    elif typ in ("sirt", "mlem", "gridrec"):
        out = dict(backprojector=typ, tomo_params={k: str(v) for k, v in s.items()})
        s = {}
    elif typ == "python":
        out = dict(backprojector="external", external_bp=dict(s))
        s = {}
    else:
        raise ValueError(f"unknown backprojector '{typ}' (ghbp, fbp, sirt, mlem, gridrec, python)")
    if s:
        raise ValueError(f"backprojector {typ}: unknown parameters {sorted(s)}")
    return out


def angle_subset_config(spec):
    """params["angle_subset"] ("" or "I/N") -> (subset, subsets)"""
    text = str(spec or "").strip()
    if not text:
        return 0, 1
    try:
        i, n = (int(v) for v in text.split("/"))
    except ValueError:
        raise ValueError(f"angle_subset must be 'I/N' (subset I of N), not {spec!r}") from None
    if not 0 <= i < n:
        raise ValueError(f"angle_subset {spec!r}: I must be in [0, N)")
    return i, n


@dataclass
class SliceInfo:
    fraction: float
    view_height: int
    nabu_index: int
    height_mm: float


@dataclass
class HelicalRecon:
    projections_dir: str
    cache_dir: str | None = None
    device: str = "cuda"
    params: dict = field(default_factory=dict)
    log: object = print

    def __post_init__(self):
        scans = bliss.discover_scans(self.projections_dir)
        pair = bliss.pair_companions(scans)
        self.main, self.companion = pair["main"], pair["companion"]
        self.log(f"scans: {self.main.name} ({len(self.main.unwrapped_angles_deg)} projections) + "
                 f"{self.companion.name} ({len(self.companion.unwrapped_angles_deg)})")
        for s in (self.main, self.companion):
            s.check_files()
        given = dict(self.params)
        self.params = {**DEFAULTS, "filters": [dict(f) for f in DEFAULTS["filters"]]}
        self._update(given)   # (validates the mode before the calibration)
        self._calibrate()
        self._build()

    # ------------------------------------------------------------------ setup
    def _calibrate(self):
        """The geometry calibration (params["geometry"]: given, cached or made), the poses it gives the frames, the
        merged frame laid out from them, the intensity calibration in that frame (calibration.calibrate) and the
        reconstruction's terms of both scans (geometry.reconstruction_terms)."""
        from . import geometry as G
        scans = (self.main, self.companion)
        path, raw = self._geometry_file()
        cal = json.loads(raw)
        for k, s in enumerate(scans):
            if os.path.basename(str(cal["scans"][k])) != os.path.basename(str(s.master)):
                raise ValueError(f"geometry {path} was made from {cal['scans'][k]}, not from {s.master} (first scan: the one with the rotation axis in its field)")
        sign = cal["layout"]["row_sign"]
        H = [sign * np.asarray(s.z_translation_m, np.float64) / s.pixel_size_m for s in scans]
        poses, not_applied = G.frame_poses(cal, [np.asarray(s.unwrapped_angles_deg, np.float64) for s in scans], H)
        layout = G.merged_layout(cal, poses, H)
        self.cal, self.arrays = calibration.calibrate(self.main, self.companion, layout, cache_dir=self.cache_dir, log=self.log)
        first_helix = cal["layout"]["frames"][0]["rest1"]
        origin = float(H[0][first_helix] - H[0][0])      # the slices keep their numbers: the helix's first frame lies where the record puts it
        terms = {}
        for s, pose, h in zip(scans, poses, H):
            pl = self.cal.placement[s.name]
            terms[s.name] = G.reconstruction_terms(pose, h, cal["layout"], rot_center=float(self.cal.rot_center), column_offset=pl.shift_x - pl.x_readjustment,
                                                   z_readjustment=pl.z_readjustment, row_origin=origin)
        m = cal["map"]
        self._geometry_data = dict(cal=cal, terms=terms, not_applied=not_applied, layout=layout, rolling_shutter=G.rolling_shutter_of(cal),
                                   map=dict(map_i=[int(x) for x in m["i"]], map_j=[int(x) for x in m["j"]], map_eu=[float(x) for x in m["eu"]], map_ev=[float(x) for x in m["ev"]],
                                            map_c0=float(m["c0"]), map_cs=float(m["cs"]), map_v0=float(m["v0"]), map_vs=float(m["vs"])),
                                   path=path, sha256=hashlib.sha256(raw.encode()).hexdigest())
        self._geometry_key = self._geometry_request()

    def _geometry_request(self):
        """what names the geometry calibration in use: the parameters, and the given file's state"""
        path = str(self.params.get("geometry") or "").strip()
        return (path, os.path.getmtime(path) if path else None, str(self.params.get("row_time_us") or "").strip())

    def _geometry_file(self):
        """(path, text) of the geometry calibration: params["geometry"], or geometry.json of the cache directory, made
        by the geometry calibration step (geometry.calibrate_geometry) when it is not there or is other scans'"""
        from . import geometry as G
        path = str(self.params.get("geometry") or "").strip()
        if path:
            with open(path) as fh:
                return os.path.abspath(path), fh.read()
        if not self.cache_dir:
            raise ValueError("no geometry calibration: give geometry=PATH (made by helirec-calibration geometry), or a cache directory "
                             "to make it in and keep it")
        path = os.path.join(self.cache_dir, "geometry.json")
        masters = [str(s.master) for s in (self.main, self.companion)]
        if os.path.exists(path):
            with open(path) as fh:
                raw = fh.read()
            try:
                made = [os.path.basename(str(x)) for x in json.loads(raw)["scans"]]
            except (ValueError, KeyError, TypeError):
                made = None
            if made == [os.path.basename(x) for x in masters]:
                self.log(f"geometry calibration loaded from {path}")
                return os.path.abspath(path), raw
            self.log(f"geometry calibration {path}: other scans or unreadable, made again")
        rt = str(self.params.get("row_time_us") or "").strip()
        self.log("geometry calibration of the scans (helirec-calibration geometry: layout, rows' timing, detector map, poses, motion) ...")
        cal = G.calibrate_geometry(masters, None if not rt else float(rt) * 1e-6, device=self.device, log=self.log)
        os.makedirs(self.cache_dir, exist_ok=True)
        G.save(cal, path)
        with open(path) as fh:
            return os.path.abspath(path), fh.read()

    def _geometry(self):
        """the geometry calibration in use, with the reconstruction's terms of both scans in the merged frame"""
        return self._geometry_data

    def _scan_data(self, scan, tag):
        pl = self.cal.placement[scan.name]
        reader = _core.RawReader(list(scan.projections.files), scan.projections.dataset, *scan.frame_shape)
        flats = (self.arrays[f"{tag}_flats"] - self.arrays[f"{tag}_dark"][None]).astype(np.float32)
        if self.legacy:   # the reference: the recorded angles and heights, one axis and one vector (the merged frame)
            return _core.make_scan_data(
                np.ascontiguousarray(scan.unwrapped_angles_deg, np.float64),
                np.ascontiguousarray(scan.z_translation_m, np.float64), scan.pixel_size_m, scan.frame_shape[0],
                scan.frame_shape[1], pl.shift_x, pl.x_readjustment, pl.z_readjustment,
                np.ascontiguousarray(scan.control_A, np.float64),
                np.ascontiguousarray(scan.flats_unwrapped_angles_deg, np.float32),
                np.ascontiguousarray(self.arrays[f"{tag}_dark"]), np.ascontiguousarray(flats),
                np.ascontiguousarray(self.arrays[f"{tag}_weight"]), reader, scan.projections.file_index,
                scan.projections.frame_index, 1.0)
        geo = self._geometry()
        t = geo["terms"][scan.name]
        return _core.make_scan_data(
            np.ascontiguousarray(t["angles_deg"], np.float64),
            np.ascontiguousarray(scan.z_translation_m, np.float64), scan.pixel_size_m, scan.frame_shape[0],
            scan.frame_shape[1], pl.shift_x, pl.x_readjustment, pl.z_readjustment,
            np.ascontiguousarray(scan.control_A, np.float64),
            np.ascontiguousarray(scan.flats_unwrapped_angles_deg, np.float32),
            np.ascontiguousarray(self.arrays[f"{tag}_dark"]), np.ascontiguousarray(flats),
            np.ascontiguousarray(self.arrays[f"{tag}_weight"]), reader, scan.projections.file_index,
            scan.projections.frame_index, t["z_translation_scale"],
            row_offset=t["row_offset"].tolist(), row_slope=t["row_slope"].tolist(), slope_col=t["slope_col"],
            col_offset=t["col_offset"].tolist(), col_slope=t["col_slope"].tolist(), slope_row=t["slope_row"],
            col_scale=t["col_scale"].tolist(), scale_col=t["scale_col"], row_scale=t["row_scale"].tolist(), scale_row=t["scale_row"],
            **geo["map"])

    def rot_center(self) -> float:
        """the rotation axis in the merged frame (the frame's own: every projection's axis is in its column offset)"""
        return float(self.cal.rot_center)

    def rolling_shutter(self) -> float:
        """the geometry calibration's row timing, as the time from the first stored row to the last in frame periods
        (0 in mode legacy_reference: one angle per frame)"""
        return 0.0 if self.legacy else float(self._geometry()["rolling_shutter"])

    def _build(self):
        self.rec = self._new_reconstructor()
        self.info = self.rec.info
        self.lo, self.hi = self.info["doable"]
        if self.info["angle_interp"] == "views" and self.exposure_arc() != 0:
            self.log("note: exposure_arc is disabled with angle_interp=views (not deconvolved; angle_interp=nearest applies it)")

    def _new_reconstructor(self):
        """a SliceReconstructor of the current params, arrays and calibration"""
        p = dict(self.params)
        m = self.main
        mode = p.pop("mode")
        cfg = dict(p, device=self.device, energy_kev=m.energy_kev, distance_m=m.distance_m,
                   pixel_size_m=m.pixel_size_m, rot_center=self.rot_center(),
                   legacy_reference=mode == "legacy_reference")
        cfg.update(backprojector_config(p["backprojector"]))
        if not cfg.get("grid_size"):
            # the slice grid of the merged frame's axis (the reconstructor's rule, 2 x axis rounded to even, + 1)
            cfg["grid_size"] = int(round(2.0 * float(self.cal.rot_center))) + 1
        self._resolved_sld = sld_config(cfg.pop("sld"))
        cfg.update(self._resolved_sld)
        cfg["exposure_arc"] = self.exposure_arc()
        cfg["sharpness_weighting"] = "" if mode == "legacy_reference" else self.sharpness_weighting()
        cfg["angle_subset"], cfg["angle_subsets"] = angle_subset_config(cfg.pop("angle_subset"))
        for key in CALIBRATION_KEYS:     # (of the scans: _scan_data)
            cfg.pop(key)
        cfg["rolling_shutter"] = self.rolling_shutter()
        scans = [self._scan_data(self.main, "main"), self._scan_data(self.companion, "companion")]
        return _core.SliceReconstructor(scans, cfg)

    def exposure_arc(self) -> float:
        """params["exposure_arc"] resolved: a number, or "auto" = the main scan's recorded fraction (0 if unknown)"""
        v = self.params["exposure_arc"]
        if isinstance(v, str) and v.strip().lower() == "auto":
            return float(self.main.exposure_arc or 0.0)
        return float(v)

    def sharpness_weighting(self) -> str:
        """params["sharpness_weighting"] resolved to "a11:a12:a21:a22:c1:c2" or "" (off); "auto" = the calibration's"""
        v = self.params["sharpness_weighting"]
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "off", "none")):
            return ""
        if isinstance(v, str) and v.strip().lower() == "auto":
            field = getattr(self.cal, "detector_blur_field", None) or []
            return ":".join(f"{x:.6g}" for x in field)
        if not isinstance(v, str):
            v = ":".join(f"{float(x):.6g}" for x in v)
        if len(v.split(":")) != 6:
            raise ValueError("sharpness_weighting must be 'auto', 'off' or a11:a12:a21:a22:c1:c2")
        return v

    @property
    def legacy(self) -> bool:
        return self.params["mode"] == "legacy_reference"

    def _update(self, kw):
        if "detector_smear" in kw:     # (the name of sharpness_weighting until 2026-11-11: configs written before)
            kw = dict(kw)
            kw.setdefault("sharpness_weighting", kw.pop("detector_smear"))
        gone = sorted(set(kw) & set(REMOVED))
        if gone:
            raise KeyError(f"{gone}: removed with the calibration they belonged to; the geometry calibration "
                           f"(params['geometry'], helirec-calibration geometry) holds what they stood for")
        regular = {k: v for k, v in kw.items() if not is_filter_key(k) or k == "filters"}
        unknown = set(regular) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}; known: {sorted(DEFAULTS)} and TYPE.param filter keys")
        mode = regular.get("mode", self.params["mode"])
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        named = {k: regular.get(k, self.params[k]) for k in CALIBRATION_KEYS}    # (the calibration in use: kept by every mode)
        if mode == "legacy_reference":
            # no partial legacy: the reference's parameters are pinned; anything else is an error
            bad = sorted(k for k, v in regular.items()
                         if k != "mode" and k not in CALIBRATION_KEYS and not _same(k, v, LEGACY_REFERENCE[k]))
            bad += sorted(k for k in kw if is_filter_key(k) and k != "filters")
            if bad:
                raise ValueError(f"mode legacy_reference pins {bad} to the reference values "
                                 f"({ {k: LEGACY_REFERENCE[k] for k in bad if k in LEGACY_REFERENCE} }); vary them in "
                                 f"mode 'corrected' instead")
            self.params = {"mode": mode, **LEGACY_REFERENCE, "filters": [dict(f) for f in LEGACY_REFERENCE["filters"]], **named}
            return
        if self.params["mode"] == "legacy_reference":   # leaving the legacy mode: start again from the defaults
            self.params = {**DEFAULTS, "filters": [dict(f) for f in DEFAULTS["filters"]], **named}
        self.params.update(regular)
        changes = {k: v for k, v in kw.items() if is_filter_key(k) and k != "filters"}
        if changes:
            self.params["filters"] = override(self.params["filters"], changes)

    def provenance(self) -> dict:
        """Everything that defines the reconstruction: mode and its conventions, resolved parameters, the intensity
        calibration and the merged frame, the geometry calibration, physics and the build. Parameters are a snapshot
        so later method/filter changes cannot alter saved provenance."""
        mode = self.params["mode"]
        return dict(mode=mode, conventions=CONVENTIONS[mode], params=copy.deepcopy(self.params),
                    resolved_sld=copy.deepcopy(getattr(self, "_resolved_sld", {})),
                    pinned=sorted(LEGACY_REFERENCE) if mode == "legacy_reference" else [],
                    output_units=CONVENTIONS[mode]["output_units"],
                    calibration=dict(fingerprint=self.cal.fingerprint, version=self.cal.version, rot_center=self.cal.rot_center,
                                     vector=list(self.cal.vector), cor_main_detector=self.cal.cor_main_detector),
                    resolved_sharpness_weighting=self.sharpness_weighting(),
                    resolved_rot_center=self.rot_center(), resolved_rolling_shutter=self.rolling_shutter(),
                    geometry=self._geometry_provenance(),
                    reconstructor=dict(self.info), pixel_size_m=self.main.pixel_size_m,
                    energy_kev=self.main.energy_kev, distance_m=self.main.distance_m,
                    helirec=_build_identity())

    def _geometry_provenance(self):
        geo = self._geometry()
        cal = geo["cal"]
        return dict(path=geo["path"], sha256=geo["sha256"], scans=cal["scans"], map={k: cal["map"][k] for k in ("c0", "cs", "v0", "vs", "i", "j", "eu", "ev")},
                    row_time_s_per_row=cal["row_time"]["seconds_per_row"], merged_frame=copy.deepcopy(geo["layout"]),
                    applied="the merged frame alone (mode legacy_reference)" if self.legacy else "the detector map and every projection's pose and row timing",
                    measured_not_applied=copy.deepcopy(geo["not_applied"]))

    def set_params(self, **kw):
        """Change reconstruction parameters (rebuilds the reconstructor; calibration is kept). Filter chain:
        filters=[...] or dotted keys, e.g. set_params(**{"paganin.delta_beta": 500}). mode="legacy_reference" pins
        every other parameter (see the module docstring)."""
        self._update(kw)
        if self._geometry_request() != self._geometry_key:   # another geometry calibration: the merged frame and the terms again
            self._calibrate()
        self._build()

    def chains(self, sweep):
        """Override dicts ({"paganin.delta_beta": 500, ...}) -> filter chains based on the current one."""
        if self.legacy and any(sweep):
            raise ValueError("mode legacy_reference: the filter chain is pinned (sweeps belong in mode 'corrected')")
        return [override(self.params["filters"], s) for s in sweep]

    # ------------------------------------------------------------------ slice addressing
    def slice_info(self, fraction: float) -> SliceInfo:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        k = int(round(fraction * (self.hi - self.lo)))     # python round: half to even (reference)
        h = self.hi - 1 - k
        z0 = float(self.main.z_translation_m[0])
        px_mm = self.main.pixel_size_m * 1e3
        rows = self.main.frame_shape[0]
        height = px_mm * ((rows - 1) / 2 - h) - z0 * 1e3
        return SliceInfo(fraction, h, 999999 - (self.hi - h), height)

    def reconstruct_heights(self, h_lo: int, h_hi: int, debug=False, timing=False, roi=None, next=None, uint8=None,
                            float32=True, progress=None, mask=None, region=None, float32_to=None, uint8_to=None, z_to=0):
        """Slices [h_lo, h_hi) -> (n, N, N), or (n, y1 - y0, x1 - x0) with roi = (x0, x1, y0, y1).
        next = (h_lo, h_hi): the chunk the caller will ask for next; its raw data is read while this one computes
        (CUDA).
        uint8 = (lo, hi): also the uint8 slices of that window (0 outside the reconstruction disk when the outer
        circle is clipped), converted on the device -> (float32, uint8), or the uint8 slices alone with float32=False.
        progress: a helirec.ProgressBar advanced by one per slice, its postfix the step in hand.
        mask: the uint8 slices are 0 where it is 0: a uint8 array of the output's shape, or a helirec OutputMask
        (possibly coarser: ReferenceMask.read, the mask of a masked reference volume).
        region = (x0, x1, y0, y1): only these pixels of the slice are computed; the rest of the output is 0. With
        uint8 slices alone and a mask, only the pixels within the bounds of what the mask keeps are computed.
        float32_to / uint8_to: an OmeZarrWriter that takes the float32 / uint8 slices as slices [z_to, z_to + n) of its
        volume, straight from the reconstructor (each slice as the block that was computed: no whole slice is built);
        None is returned in place of that array."""
        if uint8 is not None:   # (lo, hi), or (lo, hi, zero_outside) to say it outright
            uint8 = (float(uint8[0]), float(uint8[1]),
                     bool(uint8[2] if len(uint8) > 2 else self.params["clip_outer_circle"]))
        return self.rec.reconstruct(int(h_lo), int(h_hi), debug, timing, None if roi is None else tuple(map(int, roi)),
                                    None if next is None else (int(next[0]), int(next[1])), uint8, float32, progress, mask,
                                    None if region is None else tuple(map(int, region)), float32_to, uint8_to, int(z_to))

    def reconstruct_fractions(self, fractions):
        out = []
        for f in fractions:
            si = self.slice_info(f)
            out.append(self.reconstruct_heights(si.view_height, si.view_height + 1)[0])
        return np.stack(out)


def _build_identity():
    """helirec source revision (git) and extension build, for provenance"""
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = dict(extension=os.path.abspath(_core.__file__))
    try:
        out["git"] = subprocess.run(["git", "-C", root, "describe", "--always", "--dirty"], capture_output=True,
                                    text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 (no git)
        pass
    try:
        out["extension_mtime"] = os.path.getmtime(_core.__file__)
        import hashlib
        from pathlib import Path
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")) + sorted(Path(__file__).parent.rglob("*.h")):
            digest.update(str(path.relative_to(Path(__file__).parent)).encode())
            digest.update(path.read_bytes())
        out["python_source_sha256"] = digest.hexdigest()
    except OSError:
        pass
    return out
