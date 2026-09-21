"""Self-contained, measured-arclength display artifacts for VC3D Constraint Graph.

The binary payload is little-endian float64, five values per grid vertex:
XYZ in graph voxels followed by UV in display units. Patch grids retain original
vertices and an explicit accepted-quad mask. Fiber centerlines retain
every accepted sample; the two extra rows are synthetic display ribbons.
"""
import json
import os
from pathlib import Path
import uuid
import shutil

import numpy as np

from winding_arclength import apply_arclength, measure_arclength
from winding_topology import patch_corners, quad_runs


def use_measured_arclength(document, binary):
    """Convert a newly written angular payload before publishing its manifest."""
    layout = document['layout']
    if layout.get('arclength'):
        raise ValueError('Workspace already uses measured arclength')
    payload = np.memmap(binary, dtype='<f8', mode='r+').reshape(-1, 5)
    width = layout['turn_width']
    profile = measure_arclength(payload, document['sources'], width, layout['bounds'])
    bounds = apply_arclength(payload, document['sources'], width, profile)
    payload.flush()
    del payload
    spacing = layout['spacing']
    while np.prod(np.ceil((np.array(bounds[1]) - bounds[0]) / spacing) + 4) > 16_000_000:
        spacing *= 2
    layout.update(arclength=profile, bounds=bounds, raster_spacing=spacing,
                  horizontal_axis='integrated measured contour arclength',
                  angular_initialization_turn_width=layout.pop('turn_width'))
    document['settings']['spacing'] = spacing


def use_straightening(document, binary, row_band=40., progress=print, evidence=None,
                      slim_executable=None, control_spacing=100., iterations=50, work_dir=None, drop_folded_quads=False):
    """Publish a metric-aware SLIM placement after fine-geometry validation."""
    from winding_slim import fit_workspace
    payload = np.memmap(binary, dtype='<f8', mode='r+').reshape(-1, 5)
    charts, uv, report = fit_workspace(payload, document['sources'], evidence=evidence,
        row_band=row_band, spacing=control_spacing, iterations=iterations,
        executable=slim_executable, progress=progress,
        initialize_from_saved=bool(document['layout'].get('straightening')),
        overlap_distance=document['settings'].get('radius',1.),work_dir=work_dir,drop_folded_quads=drop_folded_quads)
    install_straightened_solution(document, payload, charts, uv, report, progress=progress)
    payload.flush()
    del payload


def install_straightened_solution(document, payload, charts, uv, report, progress=print):
    """Install a validated solution into an unpublished candidate payload."""
    from winding_slim import apply_solution
    if not report.get('validation', {}).get('accepted', False):
        raise ValueError('Straightening rejected; original workspace preserved')
    bounds = apply_solution(charts, payload, uv, progress=progress)
    layout = document['layout']
    spacing = layout['spacing']
    while np.prod(np.ceil((np.array(bounds[1])-bounds[0])/spacing)+4) > 16_000_000:
        spacing *= 2
    layout.update(straightening=report, bounds=bounds, raster_spacing=spacing,
                  horizontal_axis='SLIM intrinsic metric with constant vertical-fiber columns',
                  height_axis=('intrinsic SLIM coordinates with a soft source-Z row reference'
                               if report.get('row_reference') == 'source_z' else
                               'measured vertical arclength with a soft source-Z row reference'))
    document['settings']['spacing'] = spacing


def convert_workspace(path, output, *, straighten=False, row_band=40., growth_report=None,
                      slim_executable=None, control_spacing=100., iterations=50, work_dir=None, drop_folded_quads=False):
    """Regenerate a legacy workspace without rerunning growth or changing XYZ."""
    path, output = Path(path), Path(output)
    document = json.loads(path.read_text())
    if (document.get('type') != 'vc3d_winding_layout' or document.get('version') not in (1, 2)
            or document.get('geometry_format') != 'little_endian_float64_xyzuv'):
        raise ValueError('Unsupported winding workspace')
    if document['layout'].get('arclength') and not straighten:
        raise ValueError('Workspace already uses measured arclength')
    name = Path(document['geometry_file'])
    if name.is_absolute() or name.name != str(name):
        raise ValueError('Workspace geometry must be a sibling file')
    original = path.parent / name
    consumed = 0
    for source in document['sources']:
        if (source['rows'] < 1 or source['cols'] < 1 or source['byte_offset'] != consumed
                or (source['fiber'] and (source['rows'] != 3 or source['cols'] < 2))):
            raise ValueError('Invalid workspace source dimensions or range')
        consumed += source['rows'] * source['cols'] * 40
    if original.stat().st_size != document['geometry_bytes'] or consumed != document['geometry_bytes']:
        raise ValueError('Workspace geometry size mismatch')
    binary = output.with_name(output.stem + '.' + uuid.uuid4().hex + '.bin')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        shutil.copyfile(original, binary)
        if not document['layout'].get('arclength'):
            use_measured_arclength(document, binary)
        if straighten:
            reference = document['layout'].get('report_file',path.name.removesuffix('.workspace.json')+'.json')
            report_path = Path(growth_report) if growth_report else path.parent/reference
            if report_path.is_file():document['layout']['report_file']=str(report_path.resolve())
            evidence = json.loads(report_path.read_text())['evidence'] if report_path.is_file() else None
            use_straightening(document, binary, row_band=row_band, evidence=evidence,
                             slim_executable=slim_executable, control_spacing=control_spacing,
                             iterations=iterations, work_dir=work_dir, drop_folded_quads=drop_folded_quads)
        document['geometry_file'] = binary.name
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        os.replace(temporary, output)
    except BaseException:
        binary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    return output


def workspace_settings(raw):
    defaults = dict(turn_width=4096.0, spacing=20.0, fiber_half_width=20.0)
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError('workspace accepts turn_width, spacing, and fiber_half_width')
    defaults.update(raw)
    for key, value in defaults.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'workspace.{key} must be finite and positive')
    return defaults


def fiber_ribbon(member, offset, frame, turn_width, half_width):
    zyx = member['zyx'].astype(np.float64)
    uv = np.column_stack(((member['theta'] + member['lift'] + offset) * turn_width, zyx[:, 0]))
    tangent = np.gradient(uv, axis=0)
    lateral_uv = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    norm = np.linalg.norm(lateral_uv, axis=1)
    # Duplicate samples borrow the nearest usable tangent without changing
    # the centerline or connecting separate clipped runs.
    good = np.flatnonzero(norm > 1e-10)
    if not len(good):
        raise ValueError(f'fiber has no theta/z extent: {member["key"]}')
    bad = np.flatnonzero(norm <= 1e-10)
    if len(bad):
        nearest = np.searchsorted(good, bad).clip(max=len(good) - 1)
        lateral_uv[bad] = lateral_uv[good[nearest]]
        norm[bad] = norm[good[nearest]]
    lateral_uv *= half_width / norm[:, None]
    # Local inverse of (unwrapped theta, source z), avoiding any assumption
    # about fiber orientation or a circular scroll in source coordinates.
    gradient = np.empty_like(zyx)
    for axis in range(3):
        step = np.zeros(3); step[axis] = .25
        delta = frame(zyx + step) - frame(zyx - step)
        gradient[:, axis] = ((delta + .5) % 1 - .5) * (turn_width / .5)
    denom = np.sum(gradient[:, 1:] ** 2, axis=1)
    if np.any(denom < 1e-12):
        raise ValueError(f'fiber ribbon has undefined angular direction: {member["key"]}')
    lateral = np.empty_like(zyx)
    lateral[:, 0] = lateral_uv[:, 1]
    lateral[:, 1:] = gradient[:, 1:] * (
        (lateral_uv[:, 0] - gradient[:, 0] * lateral[:, 0]) / denom)[:, None]
    xyz = np.stack((zyx - lateral, zyx, zyx + lateral))[:, :, ::-1]
    placed = np.stack((uv - lateral_uv, uv, uv + lateral_uv))
    return np.concatenate((xyz, placed), axis=2)


def export_workspace(report_path, members, fields, offsets, seed, frame, raw):
    settings = workspace_settings(raw.get('workspace', {}))
    output = Path(report_path).with_suffix('.workspace.json')
    binary = output.with_name(output.stem + '.' + uuid.uuid4().hex + '.bin')
    temporary = output.with_name(output.name + '.tmp')
    sources = []
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    seed_id = None
    try:
        with binary.open('wb') as stream:
            def write_source(data, **source):
                nonlocal lo, hi
                data = np.asarray(data, dtype='<f8')
                if not np.isfinite(data).all():
                    raise ValueError(f'nonfinite workspace geometry: {source["id"]}')
                valid = data[..., 0] != -1
                if not valid.any():
                    raise ValueError(f'empty workspace source: {source["id"]}')
                lo = np.minimum(lo, data[..., 3:][valid].min(axis=0))
                hi = np.maximum(hi, data[..., 3:][valid].max(axis=0))
                sources.append(dict(source, rows=data.shape[0], cols=data.shape[1], byte_offset=stream.tell()))
                stream.write(data.tobytes(order='C'))

            for member in members.values():
                if member['node'] not in offsets:
                    continue
                key = 'fiber:' + member['key']
                if member['node'] == seed and seed_id is None:
                    seed_id = key
                data = fiber_ribbon(member, offsets[member['node']], frame,
                                    settings['turn_width'], settings['fiber_half_width'])
                write_source(data, id=key, fiber=True, component=member['node'],
                             path=member['source_file'] or '', first_point_id=int(member['ids'][0]),
                             last_point_id=int(member['ids'][-1]))
            for pid, field in fields.fields.items():
                original = fields.patches[pid].zyxs.cpu().numpy()[..., ::-1]
                for label in np.unique(field['labels']):
                    node = fields.key(pid, label)
                    if node not in offsets:
                        continue
                    selected = np.flatnonzero(field['labels'] == label)
                    r = field['graph'].node_qi[selected]; c = field['graph'].node_qj[selected]
                    data, mask, (r0, c0) = patch_corners(original, r, c)
                    valid = data[..., 0] != -1
                    theta = np.zeros(valid.shape)
                    theta[valid] = frame(data[valid, :3][:, ::-1])
                    lifts = np.full(valid.shape, np.nan)
                    center_turn = field['theta'][selected] + field['lift'][selected] + offsets[node]
                    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        rr, cc = r-r0+dr, c-c0+dc
                        proposed = np.rint(center_turn - theta[rr, cc])
                        old = lifts[rr, cc]
                        if np.any(np.isfinite(old) & (old != proposed)):
                            raise ValueError(f'Inconsistent winding at shared source corner: {node}')
                        lifts[rr, cc] = proposed
                    data[valid, 3] = (theta[valid] + lifts[valid]) * settings['turn_width']
                    data[valid, 4] = data[valid, 2]
                    write_source(data, id=node, fiber=False, component=node,
                                 path=str(Path(raw['patches_dir']).resolve() / pid),
                                 source_ij_origin=[r0, c0], quad_runs=quad_runs(mask))
        if seed_id is None:
            raise ValueError('workspace seed has no exported fiber')
        # Match the viewer's bounded raster allocation. Angular UV is only an
        # intermediate coordinate; the published payload uses measured length.
        spacing = settings['spacing']
        while np.prod(np.ceil((hi - lo) / spacing) + 4) > 16_000_000:
            spacing *= 2
        document = dict(type='vc3d_winding_layout', version=2, seed=seed_id,
            geometry_file=binary.name, geometry_bytes=binary.stat().st_size,
            geometry_format='little_endian_float64_xyzuv', sources=sources,
            settings=dict(spacing=spacing, radius=raw.get('growth', {}).get('overlap_distance', 1.0),
                fiber_scale=.25, patch_scale=1., normal_scale=4., volume_scale=1., stiffness=1.,
                normal_manifest='', fiber_folder=str(Path(raw['fibers']).resolve()),
                patch_folder=str(Path(raw['patches_dir']).resolve())),
            layout=dict(**settings, raster_spacing=spacing, height_axis='source_z',
                horizontal_axis='turn_width * (theta_fraction + turns)', bounds=[lo.tolist(), hi.tolist()],
                report_file=Path(report_path).name, read_only=True,
                patch_samples='original quad vertices with explicit accepted faces', fiber_support='synthetic display ribbons'))
        try:
            use_measured_arclength(document, binary)
        except ValueError as error:
            if (any(not s['fiber'] for s in sources)
                    or str(error) != 'No measured horizontal arclength in accepted geometry'):
                raise
            # A standalone vertical fiber supplies no horizontal metric. Keep
            # its angular chart explicit rather than dropping the component or
            # inventing a measured contour width.
            document['layout']['arclength_unavailable'] = str(error)
            document['layout']['horizontal_axis'] = 'turn_width * unwrapped theta (horizontal metric unavailable)'
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        os.replace(temporary, output)
    except BaseException:
        binary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replace fixed-width winding coordinates with measured arclength')
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--straighten-verticals', action='store_true',
                        help='Fit shared vertical columns and soft Z rows; reject folds or worse sampled metric')
    parser.add_argument('--row-band', type=float, default=40.,
                        help='Scale of the weak shared source-Z row prior (default: 40)' )
    parser.add_argument('--growth-report', type=Path, help='Accepted graph contacts (default: sibling growth JSON)')
    parser.add_argument('--slim-executable', type=Path, help='Workspace SLIM/Pastix driver')
    parser.add_argument('--control-spacing', type=float, default=100., help='Coarse controls in graph voxels')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--drop-folded-quads', action='store_true',
                        help='Omit quads with flipped rendered triangles from the candidate face mask')
    parser.add_argument('--work-dir', type=Path, help='Keep solver input, output, and validation for inspection')
    args = parser.parse_args()
    print(convert_workspace(args.workspace, args.output, straighten=args.straighten_verticals,
                            row_band=args.row_band, growth_report=args.growth_report,
                            slim_executable=args.slim_executable, control_spacing=args.control_spacing,
                            iterations=args.iterations, work_dir=args.work_dir,
                            drop_folded_quads=args.drop_folded_quads), flush=True)
