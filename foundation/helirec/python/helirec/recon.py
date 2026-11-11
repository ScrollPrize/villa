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
                     be changed in this mode, and the calibration must be the measured one (no overrides).
"""
from __future__ import annotations

import copy
import math
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
    axis_source_meters=1e9, sld="", x_interp="sinc", z_interp="lanczos3", angle_interp="nearest", exposure_arc="auto",
    angle_subset="", z_translation_scale="auto", detector_smear="auto", sharp_f=0.35, sharp_band="0.15:0.30",
    companion_registration="auto", stationary_registration="auto", axis_refinement="auto", rolling_shutter="off",
)
# rolling_shutter: the time from the read-out of the detector's first row to its last, in frame periods (negative when
#   the last row is read first): the rows of a frame were recorded at different angles while the sample turned, and a
#   slice, read at one row of every frame, is backprojected at those angles. "off" (default): one angle per frame;
#   "auto": what the calibration measures (measure_rolling_shutter: two reconstructions of a slab, from the upper and
#   from the lower rows of the detector, are turned against each other by the shutter; measured on first use, a
#   minute on a GPU, then cached); a number: that value. Applied with the angles of each slice (angle_interp nearest;
#   not ghbp): info["rolling_shutter"] is what is in use. Not a default: on PHerc0343P it removes the rotation
#   between the two halves of the detector's data, and against a finer scan of the sample it gains on one side of the
#   axis and loses on the other (README).
# axis_refinement: "auto" (the rotation axis the search found on its quarter-pixel grid, plus what the scan's opposite
#   views say is left: calibration.axis_from_pairs) | "off" (the search's value) | a number of pixels to add.
# stationary_registration: "auto" (the turns recorded with the z stage at rest are placed against the helix as the
#   frames register them: a row offset per turn, calibration.stationary_from_pairs, for both scans) | "off" (the
#   recorded z as it is).
# companion_registration: "auto" (the companion's frames are placed on the main scan's as the calibration registered
#   them: a row offset and a roll (rows by the column, columns by the row), per projection:
#   calibration.measure_registration) | "off" (the calibrated vector alone)
# detector_smear: "auto" (the field the calibration measured from the flats' noise; none if nothing significant) |
#   "" or "off" | "a11:a12:a21:a22:c1:c2": the detector's smear field for the band-dependent redundancy weights
#   (pipeline.hpp); sharp_f, sharp_band = "f1:f2": the frequency the weights are taken at, and the crossover band.
# z_interp: lanczos3 | cubic | linear, the vertical interpolation of a slice's (fractional) detector row.
# angle_interp: nearest (each projection whole in one bin, slices backprojected at the measured angles) | linear (the
#   reference's two-bin angular splat). ghbp, whose plan fixes its angles, keeps the splat: info["angle_interp"]
#   records the gridding in use.
# exposure_arc: fraction of the angular step swept during an exposure, deconvolved along the angle; "auto" (default)
#   takes it from the scan (bliss.BlissScan.exposure_arc; 0 for step scans), 0 leaves the sinogram as recorded.
# z_translation_scale: detector rows the sample really moves per row of recorded z translation: the calibration of
#   the helical pitch. "auto" (default): what the calibration measured from the frames (calibration.measure_pitch:
#   projections half a turn apart are mirror images displaced by the travel; 1 where no significant error was found
#   or nothing could be measured); a number: that value (1: the recorded translations as they are). An error of a
#   fraction e displaces what the top and the bottom rows of the detector see of a slice by e x (rows between them):
#   a blur along z. Volume slice indices (z = view height - first doable height) and the place of the sample in them
#   change with it. The rotation axis and overlap searches still use the recorded translations.
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
    angle_subset="", z_translation_scale=1.0, detector_smear="", sharp_f=0.35, sharp_band="0.15:0.30",
    companion_registration="off", stationary_registration="off", axis_refinement="off", rolling_shutter="off",
)
assert set(LEGACY_REFERENCE) == set(DEFAULTS) - {"mode"}

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
                      row_position="z_interp kernel (lanczos3 by default); linear next to unmeasured rows",
                      angular_gridding="angle_interp (nearest by default: one bin, measured angles; ghbp and "
                                       "linear: two-bin splat)",
                      exposure_arc="deconvolved along the angle (the scan's recorded arc by default)"),
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
    cor_override: float | None = None          # rotation axis in the MAIN scan's detector columns
    vector_override: tuple | None = None       # (dy, dx) main -> companion
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
        self.cal, self.arrays = calibration.calibrate(self.main, self.companion, cache_dir=self.cache_dir,
                                                      log=self.log, cor_override=self.cor_override,
                                                      vector_override=self.vector_override)
        self._check_calibration()
        if self._wants_rolling_shutter() and self.cal.rolling_shutter is None:
            measure_rolling_shutter(self)      # (a calibration from before the measurement: measured once, cached)
        self._build()

    def _wants_rolling_shutter(self) -> bool:
        v = self.params["rolling_shutter"]
        return isinstance(v, str) and v.strip().lower() == "auto" and not self.legacy

    # ------------------------------------------------------------------ setup
    def _scan_data(self, scan, tag):
        pl = self.cal.placement[scan.name]
        reader = _core.RawReader(list(scan.projections.files), scan.projections.dataset, *scan.frame_shape)
        flats = (self.arrays[f"{tag}_flats"] - self.arrays[f"{tag}_dark"][None]).astype(np.float32)
        return _core.make_scan_data(
            np.ascontiguousarray(scan.unwrapped_angles_deg, np.float64),
            np.ascontiguousarray(scan.z_translation_m, np.float64), scan.pixel_size_m, scan.frame_shape[0],
            scan.frame_shape[1], pl.shift_x, pl.x_readjustment, pl.z_readjustment,
            np.ascontiguousarray(scan.control_A, np.float64),
            np.ascontiguousarray(scan.flats_unwrapped_angles_deg, np.float32),
            np.ascontiguousarray(self.arrays[f"{tag}_dark"]), np.ascontiguousarray(flats),
            np.ascontiguousarray(self.arrays[f"{tag}_weight"]), reader, scan.projections.file_index,
            scan.projections.frame_index, self.z_translation_scale(), **self._registration(scan, tag))

    def _registration(self, scan, tag) -> dict:
        """make_scan_data's registration arguments: the companion's frames on the main scan's
        (params["companion_registration"]: "auto" = what the calibration measured, "off"), and both scans' turns at
        rest against the helix (params["stationary_registration"])"""
        opt = {}
        for key in ("companion_registration", "stationary_registration"):
            v = self.params[key]
            v = v.strip().lower() if isinstance(v, str) else v
            if v not in ("auto", "off", "", None):
                raise ValueError(f"{key} must be 'auto' or 'off'")
            opt[key] = v == "auto" and self.params["mode"] != "legacy_reference"
        arr = None
        if tag == "companion" and opt["companion_registration"]:
            arr = calibration.registration_arrays(getattr(self.cal, "companion_registration", None), len(scan.projections))
            self._registered = arr is not None or getattr(self, "_registered", False)
        out = {} if arr is None else dict(arr)
        if opt["stationary_registration"]:
            rows = self._stationary_rows(scan, tag)
            if rows is not None:
                out["row_offset"] = (out["row_offset"] + rows if "row_offset" in out else rows).astype(np.float32)
                self._stationary = True
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out.items()}

    def _stationary_rows(self, scan, tag):
        """rows to add to the vertical position of each projection of `scan`: the main scan's turns at rest against
        its helix (Calibration.stationary_registration); the companion's projections take the main scan's of the same
        angle, since its own registration is on the main scan's frames"""
        rows = calibration.stationary_arrays(getattr(self.cal, "stationary_registration", None), len(self.main.projections))
        if rows is None or tag == "main":
            return rows
        am = np.asarray(self.main.unwrapped_angles_deg, np.float64)
        ac = np.asarray(scan.unwrapped_angles_deg, np.float64)
        if len(am) == len(ac) and np.allclose(am, ac, atol=0.5 * float(np.median(np.abs(np.diff(am))))):
            return rows
        order = np.argsort(am)
        i = np.clip(np.searchsorted(am[order], ac), 1, len(am) - 1)
        near = np.where(np.abs(am[order[i - 1]] - ac) <= np.abs(am[order[i]] - ac), order[i - 1], order[i])
        return rows[near]

    def rot_center(self) -> float:
        """the rotation axis in the merged frame: the calibration's, refined (params["axis_refinement"])"""
        v = self.params["axis_refinement"]
        t = v.strip().lower() if isinstance(v, str) else v
        if t in ("auto", "off", "", None):
            d = float(getattr(self.cal, "axis_refinement", None) or 0.0) if t == "auto" and not self.legacy else 0.0
        else:
            try:
                d = float(t)
            except (TypeError, ValueError):
                raise ValueError("axis_refinement must be 'auto', 'off' or a number of pixels") from None
        return float(self.cal.rot_center) + d

    def rolling_shutter(self) -> float:
        """params["rolling_shutter"] resolved: "off" = 0, "auto" = the calibration's (0 if not measured), or a
        number of frame periods"""
        v = self.params["rolling_shutter"]
        t = v.strip().lower() if isinstance(v, str) else v
        if t in ("off", "", None):
            return 0.0
        if t == "auto":
            return float(getattr(self.cal, "rolling_shutter", None) or 0.0)
        try:
            return float(t)
        except (TypeError, ValueError):
            raise ValueError("rolling_shutter must be 'off', 'auto' or a number of frame periods") from None

    def _build(self):
        self.rec = self._new_reconstructor()
        self.info = self.rec.info
        self.lo, self.hi = self.info["doable"]

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
            # the slice grid of the axis the search found (the reconstructor's rule, 2 x axis rounded to even, + 1):
            # a refinement of the axis by a fraction of a pixel must not change the volume's shape or shift it
            cfg["grid_size"] = int(round(2.0 * float(self.cal.rot_center))) + 1
        self._resolved_sld = sld_config(cfg.pop("sld"))
        cfg.update(self._resolved_sld)
        cfg["exposure_arc"] = self.exposure_arc()
        cfg["detector_smear"] = "" if mode == "legacy_reference" else self.detector_smear()
        cfg["angle_subset"], cfg["angle_subsets"] = angle_subset_config(cfg.pop("angle_subset"))
        cfg.pop("z_translation_scale")   # (of the scans: _scan_data)
        cfg.pop("companion_registration")
        cfg.pop("stationary_registration")
        cfg.pop("axis_refinement")       # (in rot_center)
        cfg["rolling_shutter"] = 0.0 if mode == "legacy_reference" else self.rolling_shutter()
        scans = [self._scan_data(self.main, "main"), self._scan_data(self.companion, "companion")]
        return _core.SliceReconstructor(scans, cfg)

    def exposure_arc(self) -> float:
        """params["exposure_arc"] resolved: a number, or "auto" = the main scan's recorded fraction (0 if unknown)"""
        v = self.params["exposure_arc"]
        if isinstance(v, str) and v.strip().lower() == "auto":
            return float(self.main.exposure_arc or 0.0)
        return float(v)

    def detector_smear(self) -> str:
        """params["detector_smear"] resolved to "a11:a12:a21:a22:c1:c2" or "" (off); "auto" = the calibration's"""
        v = self.params["detector_smear"]
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "off", "none")):
            return ""
        if isinstance(v, str) and v.strip().lower() == "auto":
            field = getattr(self.cal, "detector_smear", None) or []
            return ":".join(f"{x:.6g}" for x in field)
        if not isinstance(v, str):
            v = ":".join(f"{float(x):.6g}" for x in v)
        if len(v.split(":")) != 6:
            raise ValueError("detector_smear must be 'auto', 'off' or a11:a12:a21:a22:c1:c2")
        return v

    def z_translation_scale(self) -> float:
        """params["z_translation_scale"] resolved: a number, or "auto" = the calibration's measured pitch scale"""
        v = self.params["z_translation_scale"]
        if isinstance(v, str) and v.strip().lower() == "auto":
            return float(getattr(self.cal, "z_translation_scale", None) or 1.0)
        v = float(v)
        if not v > 0:
            raise ValueError("z_translation_scale must be positive (or 'auto')")
        return v

    @property
    def legacy(self) -> bool:
        return self.params["mode"] == "legacy_reference"

    def _check_calibration(self):
        if self.legacy and self.cal.source != "measured":
            raise ValueError("mode legacy_reference reproduces the reference reconstruction: it needs the measured "
                             "calibration (no cor / vector overrides)")

    def _update(self, kw):
        regular = {k: v for k, v in kw.items() if not is_filter_key(k) or k == "filters"}
        unknown = set(regular) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}; known: {sorted(DEFAULTS)} and TYPE.param filter keys")
        mode = regular.get("mode", self.params["mode"])
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if mode == "legacy_reference":
            # no partial legacy: the reference's parameters are pinned; anything else is an error
            bad = sorted(k for k, v in regular.items() if k != "mode" and not _same(k, v, LEGACY_REFERENCE[k]))
            bad += sorted(k for k in kw if is_filter_key(k) and k != "filters")
            if bad:
                raise ValueError(f"mode legacy_reference pins {bad} to the reference values "
                                 f"({ {k: LEGACY_REFERENCE[k] for k in bad if k in LEGACY_REFERENCE} }); vary them in "
                                 f"mode 'corrected' instead")
            self.params = {"mode": mode, **LEGACY_REFERENCE, "filters": [dict(f) for f in LEGACY_REFERENCE["filters"]]}
            return
        if self.params["mode"] == "legacy_reference":   # leaving the legacy mode: start again from the defaults
            self.params = {**DEFAULTS, "filters": [dict(f) for f in DEFAULTS["filters"]]}
        self.params.update(regular)
        changes = {k: v for k, v in kw.items() if is_filter_key(k) and k != "filters"}
        if changes:
            self.params["filters"] = override(self.params["filters"], changes)

    def provenance(self) -> dict:
        """Everything that defines the reconstruction: mode and its conventions, resolved parameters, calibration
        (source measured / override + fingerprint), geometry, physics and the build. Parameters are a snapshot
        so later method/filter changes cannot alter saved provenance."""
        mode = self.params["mode"]
        return dict(mode=mode, conventions=CONVENTIONS[mode], params=copy.deepcopy(self.params),
                    resolved_sld=copy.deepcopy(getattr(self, "_resolved_sld", {})),
                    pinned=sorted(LEGACY_REFERENCE) if mode == "legacy_reference" else [],
                    output_units=CONVENTIONS[mode]["output_units"],
                    calibration=dict(source=self.cal.source, fingerprint=self.cal.fingerprint,
                                     version=self.cal.version, rot_center=self.cal.rot_center,
                                     vector=list(self.cal.vector), cor_main_detector=self.cal.cor_main_detector,
                                     z_translation_scale=getattr(self.cal, "z_translation_scale", None),
                                     axis_refinement=getattr(self.cal, "axis_refinement", None),
                                     rolling_shutter=getattr(self.cal, "rolling_shutter", None)),
                    resolved_z_translation_scale=self.z_translation_scale(),
                    resolved_detector_smear=self.detector_smear(),
                    companion_registration=bool(getattr(self, "_registered", False)),
                    stationary_registration=bool(getattr(self, "_stationary", False)),
                    resolved_rot_center=self.rot_center(), resolved_rolling_shutter=self.rolling_shutter(),
                    reconstructor=dict(self.info), pixel_size_m=self.main.pixel_size_m,
                    energy_kev=self.main.energy_kev, distance_m=self.main.distance_m,
                    helirec=_build_identity())

    def set_params(self, **kw):
        """Change reconstruction parameters (rebuilds the reconstructor; calibration is kept). Filter chain:
        filters=[...] or dotted keys, e.g. set_params(**{"paganin.delta_beta": 500}). mode="legacy_reference" pins
        every other parameter (see the module docstring)."""
        self._update(kw)
        self._check_calibration()
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


# ---------------------------------------------------------------------- the two halves of the detector
SHUTTER_SLICES = 128          # slices of the slab the two half-turn reconstructions are made of (one layer of blocks)
SHUTTER_PROBE = 1.0           # frame periods: the second value the rotation between the halves is measured at
AXIS_PROBE = 0.2              # pixels: the second axis the translation between the halves is measured at
SHUTTER_MIN_RADIUS = 256      # pixels around the axis the main scan must see from both sides
SHUTTER_MIN_BLOCKS = 40
SHUTTER_MAX = 1.5             # frame periods: a larger |value| is taken for a failed measurement
HALF_TURN_MAX_AXIS = 0.5      # pixels: a larger correction of the axis is taken for a failed measurement


def _half_turn_field(r, h0, roi, shutter, axis):
    """How the slab reconstructed from the lower rows of the main scan's detector lies on the one from its upper
    rows, at rolling_shutter = shutter and axis_refinement = axis: the similarity fitted to the shifts of blocks
    -> dict(rotation (radians), shift (dx, dy at the rotation axis, px), scale, blocks, rms (px)), or None when
    fewer than SHUTTER_MIN_BLOCKS blocks register (then `blocks` is all the caller gets: (None, n))."""
    rows = r.main.frame_shape[0]
    v = np.arange(rows, dtype=np.float64) - 0.5 * (rows - 1)
    ramp = 0.5 * (1 + np.tanh(v / 6.0))                       # 0 over the upper rows, 1 over the lower
    vols = []
    keep_params, keep_arrays = r.params, r.arrays
    try:
        for part in (1 - ramp, ramp):
            r.params = dict(keep_params, backprojector="fbp", filters=[], sld="", detector_smear="off", angle_subset="",
                            rolling_shutter=float(shutter), axis_refinement=float(axis))
            r.arrays = dict(keep_arrays,
                            main_weight=(keep_arrays["main_weight"] * np.maximum(part, 1e-9)[:, None]).astype(np.float32),
                            companion_weight=(keep_arrays["companion_weight"] * 1e-6).astype(np.float32))
            rec = r._new_reconstructor()
            out = [rec.reconstruct(h, min(h + 32, h0 + SHUTTER_SLICES), False, False, roi, None, None, True, None, None,
                                   None, None, None, 0) for h in range(h0, h0 + SHUTTER_SLICES, 32)]
            vols.append(np.concatenate(out))
            del rec, out
    finally:
        r.params, r.arrays = keep_params, keep_arrays
    N = SHUTTER_SLICES
    bc = _core.BlockCorrelator(N, 1, 16, 6, True, 0.25, 4, r.device)
    a, b = vols
    # block centres from the rotation axis: the slice's centre, pixel (grid_size - 1) / 2 (the grid _new_reconstructor pins)
    centre = 0.5 * int(round(2.0 * float(r.cal.rot_center)))
    x0, y0 = roi[0] - centre, roi[2] - centre
    t = []
    for y in range(0, a.shape[1] - N + 1, N // 2):
        for x in range(0, a.shape[2] - N + 1, N // 2):
            m = bc.compute(np.ascontiguousarray(a[:, y:y + N, x:x + N]), np.ascontiguousarray(b[:, y:y + N, x:x + N]))
            if m["var_a"] > 0 and m["peak"] >= 0.3:
                t.append((x0 + x + 0.5 * (N - 1), y0 + y + 0.5 * (N - 1), m["shift"][2], m["shift"][1]))
    if len(t) < SHUTTER_MIN_BLOCKS:
        return None, len(t)
    t = np.array(t)
    G = np.stack([np.ones(len(t)), t[:, 0], t[:, 1]], -1)
    keep = np.ones(len(t), bool)
    for _ in range(3):   # dx = a0 + a1 x + a2 y, dy = b0 + b1 x + b2 y; blocks far off the fit are left out
        cx = np.linalg.lstsq(G[keep], t[keep, 2], rcond=None)[0]
        cy = np.linalg.lstsq(G[keep], t[keep, 3], rcond=None)[0]
        res = np.hypot(t[:, 2] - G @ cx, t[:, 3] - G @ cy)
        keep = res <= 3 * 1.4826 * np.median(res) + 0.05
    return dict(rotation=0.5 * (cy[1] - cx[2]), shift=np.array([cx[0], cy[0]]), scale=0.5 * (cx[1] + cy[2]),
                blocks=int(keep.sum()), rms=float(np.sqrt((res[keep] ** 2).mean()))), len(t)


def measure_rolling_shutter(r: "HelicalRecon"):
    """The rolling shutter of the scan's camera, from the scan (Calibration.rolling_shutter, cached with it), and
    beside it what the same measurement says is left of the rotation axis's error (recorded: a check of
    Calibration.axis_refinement, which comes from the opposite views).

    A slice of a helical scan passes over every row of the detector. Reconstructed from the upper rows alone and
    from the lower rows alone (each half a turn: complete within the radius the main scan sees from both sides of the
    axis), the two volumes must coincide. What they do instead, measured on blocks (BlockCorrelator; a similarity
    fitted to the block shifts), says what is wrong with the geometry both were made with:
      * turned against each other: the rows were recorded at different angles, a rolling shutter;
      * shifted against each other: the rotation axis is off (the two halves see it from opposite sides), or the
        detector's image is sheared along the rows (recorded, not applied);
      * of different size: the magnification depends on the row (a keystone; recorded, not applied).
    Rotation and shift are linear in the shutter and in the axis: each is measured again with a probe value
    (SHUTTER_PROBE frame periods, AXIS_PROBE pixels) and the values at which they vanish are taken; of the shift,
    the part along the direction the axis moves it. The two are separate terms of one fit, so the shutter is not
    tuned against an error of the axis (with the axis off, the shift cancels the rotation on one side of the slice
    and adds to it on the other: a shutter judged by one side alone would come out wrong).
    Nothing is applied when the main scan does not see enough of both sides of the axis, there is no helix, too few
    blocks register, or a result is implausible."""
    cal, main, log = r.cal, r.main, r.log
    res = dict(rolling_shutter=0.0, measured=None, reason=None, axis=None)
    axis0 = float(cal.axis_refinement or 0.0)
    def finish(reason=None):
        res["reason"] = reason
        cal.rolling_shutter = float(res["rolling_shutter"])
        cal.details["rolling_shutter"] = res
        if res["measured"] is None:
            log(f"rolling shutter: not measured ({reason}): one angle per frame")
        else:
            f, ax = res["field_mrad_px"], res["axis"]
            log(f"the two halves of the detector ({res['blocks']} blocks): turned {f[0]:+.3f} mrad against each other "
                f"({f[1]:+.3f} with a shutter of {SHUTTER_PROBE:g}): rolling shutter {res['measured']:+.3f} frame periods "
                f"from the first row to the last" + ("" if reason is None else f" ({reason}: not applied)")
                + f"; shifted by ({f[2]:+.3f}, {f[3]:+.3f}) px: the rotation axis {ax['residual']:+.3f} px from the "
                f"refined one, a shear of {ax['row_shear']:+.2f} px per 1000 rows; magnification {1e2 * res['scale']:+.3f} % "
                f"between the halves (these three: recorded, not applied)  [{res['seconds']:.0f}s]")
        if cal.source == "measured" and r.cache_dir:
            calibration._save_calibration(r.cache_dir, cal)
        return res
    import time
    t0 = time.time()
    rows, cols = main.frame_shape
    axis = float(cal.cor_main_detector)
    radius = (int(min(axis, cols - 1 - axis)) - 48) // 64 * 64
    if radius < SHUTTER_MIN_RADIUS:
        return finish(f"the main scan sees {max(radius, 0)} px on both sides of the axis")
    ang = np.asarray(main.unwrapped_angles_deg, np.float64)
    z = np.asarray(main.z_translation_m, np.float64) / main.pixel_size_m
    moving = np.flatnonzero(np.abs(np.diff(z)) > 0.01)
    if len(moving) < 8:
        return finish("no helix")
    per_turn = abs(z[moving[-1] + 1] - z[moving[0]]) / max(abs(ang[moving[-1] + 1] - ang[moving[0]]) / 360.0, 1e-9)
    if not 0.05 * rows <= per_turn <= rows:
        return finish(f"travel of {per_turn:.0f} rows per turn")
    keep_params = r.params
    r.params = dict(keep_params, backprojector="fbp", filters=[], sld="", detector_smear="off", angle_subset="", rolling_shutter=0.0)
    try:
        probe = r._new_reconstructor()
        info = probe.info
        del probe
    finally:
        r.params = keep_params
    if info["angle_interp"] != "nearest":
        return finish("the gridding in use takes one angle per bin")
    lo, hi = info["doable"]
    # the slab: the middle of the heights the helix alone records
    zs = (z - z[0]) * r.z_translation_scale()
    h_a, h_b = sorted((zs[moving[0]], zs[moving[-1] + 1]))
    h_lo, h_hi = max(lo, int(h_a) + rows), min(hi, int(h_b))
    if h_hi - h_lo < SHUTTER_SLICES:
        h_lo, h_hi = lo, hi
    if h_hi - h_lo < SHUTTER_SLICES:
        return finish("fewer slices than the measurement's slab")
    h0 = (h_lo + h_hi - SHUTTER_SLICES) // 2
    c = info["grid_size"] // 2
    roi = (c - radius, c + radius, c - radius, c + radius)
    f0, n0 = _half_turn_field(r, h0, roi, 0.0, axis0)
    if f0 is None:
        return finish(f"only {n0} blocks of the two half-turn reconstructions register")
    f1, _ = _half_turn_field(r, h0, roi, SHUTTER_PROBE, axis0)
    f2, _ = _half_turn_field(r, h0, roi, 0.0, axis0 + AXIS_PROBE)
    res.update(seconds=time.time() - t0, blocks=f0["blocks"], heights=[h0, h0 + SHUTTER_SLICES], roi=list(roi),
               rms_px=f0["rms"], scale=float(f0["scale"]))
    if f1 is None or abs(f1["rotation"] - f0["rotation"]) < 2e-5:
        return finish("the rotation does not answer to the shutter")
    res.update(measured=float(-f0["rotation"] * SHUTTER_PROBE / (f1["rotation"] - f0["rotation"])),
               field_mrad_px=[1e3 * f0["rotation"], 1e3 * f1["rotation"], float(f0["shift"][0]), float(f0["shift"][1])])
    # the axis: the shift's part along the direction the axis moves it; across it, a shear of the rows
    ax = dict(residual=0.0, measured=None, row_shear=0.0)
    res["axis"] = ax
    if f2 is not None:
        s = (f2["shift"] - f0["shift"]) / AXIS_PROBE
        if float(s @ s) > 0.25:
            d = float(-(f0["shift"] @ s) / (s @ s))
            perp = f0["shift"] + d * s
            k = per_turn / (2 * np.pi)                                 # rows per radian of rotation
            shear = float(np.sign(s[0] * perp[1] - s[1] * perp[0]) * np.hypot(*perp) / (np.sqrt(s @ s) * k) * 1e3)
            ax.update(measured=d, row_shear=shear, sensitivity=[float(s[0]), float(s[1])])
            if abs(d) <= HALF_TURN_MAX_AXIS:
                ax.update(residual=d)
    if abs(res["measured"]) > SHUTTER_MAX:
        return finish("implausible value")
    res["rolling_shutter"] = res["measured"]
    return finish()


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
