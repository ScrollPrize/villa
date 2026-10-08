"""Flattened strip images of a traced fiber, as in VC3D's line-annotation window (vc::lasagna buildLineViewSurfaces).

Two strips along the trace: across the sheet-side direction (lineSurface) and along the sheet normal (lineSideSlice).
The sheet normal is the learned crop frame (column u) at each decision: anchored mid-commit, interpolated with sign
continuity, smoothed (NORMAL_SIGMA) and re-orthogonalized to the smoothed trace tangent; side = normal x tangent.
Samples every STEP trace voxels (one level-0 CT voxel when a trace voxel is two CT voxels), +-HALF_WIDTH across,
wrapped into panels of PANEL trace voxels. Red ticks mark the seed; an optional reference fiber (e.g. the annotation)
is drawn where it crosses each strip: green within OVERLAY_PLANE voxels of the strip's plane, orange farther out.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at

STEP, HALF_WIDTH, PANEL = .5, 32., 512.  # image sampling (trace voxels)
ANCHOR_OFFSET, NORMAL_SIGMA, TANGENT_SIGMA = 4., 4., 2.  # frame anchors / smoothing (trace voxels)
OVERLAY_PLANE = 3.  # reference points within this distance of a strip's plane are drawn green


def learned_decision_frames(predictor, vol, path, family, held, every=8.):
    """(direction, travelled, frame) every ``every`` voxels of a saved forward path from the learned frame model,
    with the tracer's heading rule (trace_heading after each commit) and the observed path so far."""
    from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import predict_frames
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import fiber_family, trace_heading
    path = np.asarray(path, float)
    s = arclength(path)
    index = np.unique(np.searchsorted(s, np.arange(0., s[-1]+1e-9, every)).clip(0, len(path)-1))
    priors, heading = [], np.asarray(held, float)
    for i in index:
        heading = trace_heading(path[:i+1], 0, heading)
        priors.append(heading)
    frames = predict_frames(predictor, vol, [path[i] for i in index], priors, [path[:i+1] for i in index],
                            [fiber_family(family)]*len(index))
    return [(1., float(s[i]), np.asarray(f)) for i, f in zip(index, frames)]


def ribbon_frames(path, seed_index, decisions):
    """Resampled centreline, unit tangents, normals and sides; the seed's column."""
    s = arclength(path)
    s_seed = s[seed_index]
    t = np.arange(0., s[-1]+1e-9, STEP)
    centre = interp_at(path, s, t)
    tangent = gaussian_filter1d(np.gradient(centre, axis=0), TANGENT_SIGMA/STEP, axis=0, mode='nearest')
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    # Each decision's frame normal anchors the middle of the stretch it committed (forward ahead of its travelled
    # distance, backward behind it); sign-continuous anchors are interpolated along the trace and smoothed.
    keys = np.array([sign*(k+ANCHOR_OFFSET) for sign, k, _ in decisions])
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    values = np.stack([decisions[i][2][:, 0] for i in order])
    for k in range(1, len(values)):
        if values[k] @ values[k-1] < 0:
            values[k] = -values[k]
    rel = t-s_seed
    normal = np.stack([np.interp(rel, keys, values[:, c]) for c in range(3)], 1)
    normal = gaussian_filter1d(normal, NORMAL_SIGMA/STEP, axis=0, mode='nearest')
    normal -= (normal*tangent).sum(1, keepdims=True)*tangent
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    side = np.cross(normal, tangent)
    return centre, tangent, normal, side, int(round(s_seed/STEP))


def sample_strip(vol, centre, direction, block=256, half_width=HALF_WIDTH):
    """(rows, columns) uint8 CT on centre + offset*direction, offsets -half_width..half_width (trace voxels)."""
    offsets = np.arange(-half_width, half_width+1e-9, STEP)
    out = np.zeros((len(offsets), len(centre)), np.uint8)
    shape = np.asarray(vol.ct.shape)
    for a in range(0, len(centre), block):
        pts = centre[a:a+block, None]+offsets[None, :, None]*direction[a:a+block, None]  # (C, R, 3) xyz trace
        zyx = pts[..., ::-1]*vol.input_scale
        lo = np.floor(zyx.reshape(-1, 3).min(0)).astype(int)-1
        hi = np.ceil(zyx.reshape(-1, 3).max(0)).astype(int)+2
        lo_c, hi_c = lo.clip(0, shape), hi.clip(0, shape)
        if (hi_c <= lo_c).any():
            continue
        cube = np.zeros(hi-lo, np.float32)
        cube[tuple(slice(l-o, h-o) for l, h, o in zip(lo_c, hi_c, lo))] = vol.ct.read(lo_c, hi_c-lo_c)
        values = map_coordinates(cube, (zyx-lo).reshape(-1, 3).T, order=1, mode='constant', cval=0.)
        out[:, a:a+block] = values.reshape(pts.shape[:2]).T.clip(0, 255).astype(np.uint8)
    return out


def reference_marks(centre, tangent, normal, side, reference, half_width=HALF_WIDTH):
    """Reference polyline points crossing the strips: (column, side offset, normal offset), trace voxels."""
    reference = np.asarray(reference, float)
    if len(reference) > 1:
        s = arclength(reference)
        reference = interp_at(reference, s, np.arange(0., s[-1]+1e-9, STEP))
    _, column = cKDTree(centre).query(reference)
    offset = reference-centre[column]
    along, across, out = ((offset*axis[column]).sum(1) for axis in (tangent, side, normal))
    keep = (np.abs(along) <= STEP) & (np.abs(across) <= half_width) & (np.abs(out) <= half_width)
    return column[keep], across[keep], out[keep]


def render(vol, path, seed_index, decisions, title, file, reference=None, half_width=HALF_WIDTH, marks=()):
    """``marks``: (path index, RGB) ticks above and below the strips, e.g. a departure point."""
    centre, tangent, normal, side, seed_col = ribbon_frames(path, seed_index, decisions)
    strips = [sample_strip(vol, centre, side, half_width=half_width), sample_strip(vol, centre, normal, half_width=half_width)]
    data = np.concatenate([x.ravel() for x in strips])
    lo, hi = np.percentile(data[data > 0], [.5, 99.5]) if (data > 0).any() else (0, 255)
    strips = [np.repeat(((x.astype(np.float32)-lo)/max(hi-lo, 1)*255).clip(0, 255).astype(np.uint8)[..., None], 3, -1)
              for x in strips]
    if reference is not None:
        column, across, out = reference_marks(centre, tangent, normal, side, reference, half_width)
        for strip, offset, other in ((strips[0], across, out), (strips[1], out, across)):
            row = np.round((offset+half_width)/STEP).astype(int).clip(0, strip.shape[0]-1)
            near = np.abs(other) <= OVERLAY_PLANE
            strip[row[near], column[near]] = (40, 220, 40)
            strip[row[~near], column[~near]] = (255, 150, 0)
    rows, width = strips[0].shape[0], int(PANEL/STEP)
    gap, head, label = 3, 14, 12
    panels = max(1, int(np.ceil(len(centre)/width)))
    panel_h = label+2*rows+gap+6
    canvas = Image.new('RGB', (width+56, head+panels*panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 1), title, fill=(0, 0, 0))
    for p in range(panels):
        a, b = p*width, min((p+1)*width, len(centre))
        y = head+p*panel_h
        for j, strip in enumerate(strips):
            yy = y+label+j*(rows+gap)
            canvas.paste(Image.fromarray(strip[:, a:b]), (48, yy))
            draw.text((2, yy+rows//2-5), ('side', 'normal')[j], fill=(90, 90, 90))
            for x in (42, 48+b-a):  # centreline ticks outside the image
                draw.line([(x, yy+rows//2), (x+5, yy+rows//2)], fill=(0, 140, 255))
        start = (a-seed_col)*STEP
        draw.text((48, y), f'{start:+.0f} vox', fill=(90, 90, 90))
        s_path = arclength(np.asarray(path, float))
        for index, colour in marks:
            col = int(round(s_path[index]/STEP))
            if a <= col < b:
                x = 48+col-a
                draw.line([(x, y+label-4), (x, y+label-1)], fill=colour, width=2)
                draw.line([(x, y+label+2*rows+gap), (x, y+label+2*rows+gap+4)], fill=colour, width=2)
        if a <= seed_col < b:
            x = 48+seed_col-a
            draw.line([(x, y+label-4), (x, y+label-1)], fill=(230, 0, 0), width=2)
            draw.line([(x, y+label+2*rows+gap), (x, y+label+2*rows+gap+4)], fill=(230, 0, 0), width=2)
    canvas.save(file, optimize=True)
