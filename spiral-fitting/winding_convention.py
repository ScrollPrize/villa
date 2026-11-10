"""Absolute-winding anchor convention ("sheet").

The fitter compares an absolute anchor with q = shifted_radius / dr, which is a per-point BRANCH label: constant
along a sheet within one turn and stepping by one where the wrapped spiral angle phi crosses 0
(sample_spiral.get_theta_and_radii). Legacy anchor files instead carry a continuous winding W that rises by one per
turn, which disagrees with q by up to a whole winding depending on the angle.

A "sheet" anchor stores its continuous gauged winding ``w_cont`` (W plus a frozen global gauge) and is compared in
the frame of an angle phi_e (the cached crossing-map theta of its PCL node for patch rows, the point's current
angle for point rows):

    t = round(w_cont - phi_e / 2pi),   rho = w_cont - phi_e / 2pi - t

t is the target branch label; rho is a circular phase residual (the anchor's convention residual plus the fit's
angular displacement of the point, in turns). Points with |rho| > RHO_MAX are rejected: their target is too close
to the half-integer branch boundary to be trusted. The rule depends only on the final transform, never on a path:
moving a point along its own sheet by a full turn really does put it one winding further out.

Records also carry ``wind_a`` = n = round(w_cont - phi_ref / 2pi) (the label in the undeformed frame, integer >= 1)
and ``phi_ref`` = the undeformed angle, both redundant with w_cont and validated against it. Angles use the float32
representation returned by sample_spiral.get_theta, whose upper end is float32(2pi) = 6.2831854820251465.
"""
import math

import numpy as np
import torch

RHO_MAX = 0.4
PHI_MAX = float(np.float32(2 * np.pi))     # float32 % (2pi) can return this value for angles just below 2pi
CONVENTION_RESIDUAL_MAX = 0.05             # |w_cont - phi_ref/2pi - wind_a| allowed in a record
SHEET = 'sheet'
LEGACY = 'legacy'


def branch_target(w_cont, phi):
    """(t, rho) for continuous gauged windings and wrapped angles; numpy or torch, any matching shapes."""
    if isinstance(w_cont, torch.Tensor) or isinstance(phi, torch.Tensor):
        w_cont = torch.as_tensor(w_cont)
        phi = torch.as_tensor(phi, device=w_cont.device, dtype=w_cont.dtype)
        x = w_cont - phi / (2 * math.pi)
        t = torch.round(x)
        return t, x - t
    x = np.asarray(w_cont, dtype=np.float64) - np.asarray(phi, dtype=np.float64) / (2 * np.pi)
    t = np.round(x)
    return t, x - t


def accepted(rho):
    """Phase acceptance: |rho| <= RHO_MAX (inclusive)."""
    if isinstance(rho, torch.Tensor):
        return rho.abs() <= RHO_MAX
    return np.abs(rho) <= RHO_MAX


def undeformed_phi(model, zyx, chunk=200_000):
    """Wrapped undeformed spiral angle (float32 semantics, as a float64 array) of L2 zyx points.

    Uses the fitter's own slice-to-spiral transform truncated to step 0, which does not depend on the trained
    parameters (checked on three unrelated checkpoints: max difference 5e-7 rad)."""
    from sample_spiral import get_theta
    zyx = np.asarray(zyx, dtype=np.float32)
    out = np.empty(len(zyx), dtype=np.float64)
    transform = model.get_slice_to_spiral_transform(truncate_at_step=0)
    with torch.inference_mode():
        for lo in range(0, len(zyx), chunk):
            spiral = transform(torch.from_numpy(zyx[lo:lo + chunk]))
            out[lo:lo + chunk] = get_theta(spiral[:, 1:])[0].double().numpy()
    return out


def record_problems(point):
    """List of reasons a sheet-convention point dict is invalid (empty when valid).

    Expects the loaded point keys: 'winding_annotation' (wind_a), 'phi_ref', 'w_cont'."""
    problems = []
    n = point.get('winding_annotation', float('nan'))
    phi = point.get('phi_ref', float('nan'))
    w = point.get('w_cont', float('nan'))
    if not (isinstance(n, (int, float)) and math.isfinite(n) and float(n).is_integer() and n >= 1):
        problems.append(f'wind_a {n!r} is not an integer >= 1')
    if not (isinstance(phi, (int, float)) and math.isfinite(phi) and 0.0 <= phi <= PHI_MAX):
        problems.append(f'phi_ref {phi!r} outside [0, {PHI_MAX!r}]')
    if not (isinstance(w, (int, float)) and math.isfinite(w)):
        problems.append(f'w_cont {w!r} is not finite')
    if not problems and abs(w - phi / (2 * math.pi) - n) > CONVENTION_RESIDUAL_MAX:
        problems.append(f'w_cont {w!r}, phi_ref {phi!r} and wind_a {n!r} are inconsistent '
                        f'(|w_cont - phi_ref/2pi - wind_a| > {CONVENTION_RESIDUAL_MAX})')
    return problems


def is_sheet_collection(pcl):
    return (pcl.get('metadata') or {}).get('winding_convention') == SHEET


def validate_sheet_collections(point_collections, config):
    """Check every loaded collection against the configured convention; raise ValueError on any problem.

    Called right after loading (startup and live replacement), before normalisation or attachment, so a bad
    anchor file stops the run instead of being dropped. Returns (num_sheet_collections, num_sheet_points)."""
    convention = config.get('abs_winding_target_convention', LEGACY)
    at_point = bool(config.get('abs_winding_eval_at_point', False))
    errors, sheet_pcls, sheet_points, legacy_absolute = [], 0, 0, 0
    for cid, pcl in point_collections.items():
        metadata = pcl.get('metadata') or {}
        absolute = bool(metadata.get('winding_is_absolute', False))
        if not is_sheet_collection(pcl):
            legacy_absolute += absolute
            continue
        name = pcl.get('name', cid)
        if convention != SHEET:
            errors.append(f'{name}: sheet-convention collection loaded with '
                          f'abs_winding_target_convention {convention!r}')
            continue
        if not absolute:
            errors.append(f'{name}: sheet-convention collection did not enter through the absolute input role '
                          f'(input_role {metadata.get("input_role")!r})')
            continue
        sheet_pcls += 1
        for point in pcl['points'].values():
            sheet_points += 1
            problems = record_problems(point)
            if problems:
                errors.append(f'{name} point {point.get("id")}: ' + '; '.join(problems))
    if at_point and sheet_pcls and legacy_absolute:
        errors.append(f'abs_winding_eval_at_point is on but {legacy_absolute} legacy absolute collections are '
                      'loaded beside the sheet ones; mixed absolute conventions are refused in this mode')
    if errors:
        shown = '\n  '.join(errors[:20])
        more = f'\n  ... and {len(errors) - 20} more' if len(errors) > 20 else ''
        raise ValueError(f'invalid absolute winding inputs:\n  {shown}{more}')
    return sheet_pcls, sheet_points
