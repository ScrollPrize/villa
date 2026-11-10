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
# (mode "corrected") and no projection filters
DEFAULTS = dict(
    mode="corrected", filters=DEFAULT_CHAIN, margin=-1, n_grid=0, angular_tolerance_steps=5.0,
    luminosity_repartition=100.0, mend_threshold=1e-4, log_clip_min=1e-6, log_clip_max=10.0, block_angles=500,
    backprojector="ghbp", grid_size=0, ghbp_legs=8, ghbp_reduction_steps=2, clip_outer_circle=True,
    axis_source_meters=1e9, sld="", x_interp="sinc", z_interp="lanczos3", angle_interp="nearest", exposure_arc="auto",
    angle_subset="", z_translation_scale="auto", detector_smear="auto", sharp_f=0.35, sharp_band="0.15:0.30",
    companion_registration="auto",
)
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
    companion_registration="off",
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
      fbp(interp=linear|cubic|lanczos3|bspline3|bspline5, oversample=1..8)   exact pixel-driven backprojector
                                            (default interp=bspline3, oversample=2: ~ideal interpolation)
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
        self._build()

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
        (params["companion_registration"]: "auto" = what the calibration measured, "off")"""
        v = self.params["companion_registration"]
        v = v.strip().lower() if isinstance(v, str) else v
        if v not in ("auto", "off", "", None):
            raise ValueError("companion_registration must be 'auto' or 'off'")
        if tag != "companion" or v != "auto" or self.params["mode"] == "legacy_reference":
            return {}
        arr = calibration.registration_arrays(getattr(self.cal, "companion_registration", None), len(scan.projections))
        if arr is None:
            return {}
        self._registered = True
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in arr.items()}

    def _build(self):
        p = dict(self.params)
        m = self.main
        mode = p.pop("mode")
        cfg = dict(p, device=self.device, energy_kev=m.energy_kev, distance_m=m.distance_m,
                   pixel_size_m=m.pixel_size_m, rot_center=self.cal.rot_center,
                   legacy_reference=mode == "legacy_reference")
        cfg.update(backprojector_config(p["backprojector"]))
        self._resolved_sld = sld_config(cfg.pop("sld"))
        cfg.update(self._resolved_sld)
        cfg["exposure_arc"] = self.exposure_arc()
        cfg["detector_smear"] = "" if mode == "legacy_reference" else self.detector_smear()
        cfg["angle_subset"], cfg["angle_subsets"] = angle_subset_config(cfg.pop("angle_subset"))
        cfg.pop("z_translation_scale")   # (of the scans: _scan_data)
        cfg.pop("companion_registration")
        scans = [self._scan_data(self.main, "main"), self._scan_data(self.companion, "companion")]
        self.rec = _core.SliceReconstructor(scans, cfg)
        self.info = self.rec.info
        self.lo, self.hi = self.info["doable"]

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
                                     z_translation_scale=getattr(self.cal, "z_translation_scale", None)),
                    resolved_z_translation_scale=self.z_translation_scale(),
                    resolved_detector_smear=self.detector_smear(),
                    companion_registration=bool(getattr(self, "_registered", False)),
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
