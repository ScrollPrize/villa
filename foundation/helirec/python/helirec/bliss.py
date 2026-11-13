"""BLISS tomo scans (ESRF BM18) as the reconstruction reads them: helirec::calibration::Scan
(helirec/calibration/scan.hpp), derived from the master file as the reference conversion (nxtomomill h52nx followed by
nx2etf) derives it, bit for bit: the frames of the projections, darks and flat series in the detector's files
(FrameList), per projection the angle, the height and the ring current, per flat series the ring current of every
frame and an angle, and the acquisition's scalars in SI units.

    open_scan(path)                  a scan: its master file, its scan directory, or a directory holding one scan
    discover_scans(root)             the tomo scans below root, sorted by name
    pair_companions(scans)           {"main", "companion", "companions"}: the one extended-field-of-view family
    group_companions(scans)          every such family, each the main scan first
    parse_tetra_name(name)           {"base", "kind", "rank", "suffix", "crash"} of a <base>_<HA|TA..>_NNNN name, or None

Frames are read with helirec._core.RawReader(list(refs.files), refs.dataset, *scan.frame_shape).
"""
from ._core import FrameList, Scan, discover_scans, group_companions, open_scan, pair_companions, parse_tetra_name

__all__ = ["FrameList", "Scan", "open_scan", "discover_scans", "pair_companions", "group_companions", "parse_tetra_name"]
