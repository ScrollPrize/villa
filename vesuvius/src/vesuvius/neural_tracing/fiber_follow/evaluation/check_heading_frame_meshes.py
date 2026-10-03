"""Paris 4 check: verified-mesh locations, H/V seeds, frame checkpoint.

OMP_NUM_THREADS=1 python evaluation/check_heading_frame_meshes.py
OMP_NUM_THREADS=1 python evaluation/check_heading_frame_meshes.py --locations 64 --checkpoint output/heading_model_l0_w16_centered_frame/ckpt_032000.pt
Add --lasagna to compare the same sites with public Lasagna normals (downloads cached locally).
Uses the stored mesh XYZ in selected-CT coordinates (9.6 um/voxel); no mesh normal is a model input.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion

from vesuvius.tifxyz import read_tifxyz
from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config, ct_source_spec
from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.heading_model.model import load_heading_model, model_inputs, prior_frames
from vesuvius.neural_tracing.fiber_follow.heading_model.frames import family_ids, orthonormal_frame
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_seed_heading
from vesuvius.neural_tracing.fiber_follow.evaluation.compare_sheet_normals import (
    angle, stats, normal_at, open_remote, BUCKET, MANIFESTS,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=Path('output/heading_model_l0_w16_centered_frame/last.pt'))
    ap.add_argument('--locations', type=int, default=6, help='Total sites, distributed evenly across the three meshes (minimum 3).')
    ap.add_argument('--lasagna', action='store_true', help='Also evaluate public Lasagna normals at the selected sites.')
    args = ap.parse_args()
    if args.locations < 3:
        ap.error('--locations must be at least 3')
    torch.set_num_threads(1)
    checkpoint = args.checkpoint
    model, ckpt = load_heading_model(checkpoint)
    document, _ = read_dataset_config('configs/mixed_ct_datasets_paris50.json')
    source = next(s for s in document['sources'] if s['name'] == 'paris4')
    spec = ct_source_spec(source, document['cache_dir'])
    spec.ct_normalization = ckpt['ct_normalization']['volumes'][volume_key(spec)]
    vol = FiberVolume(spec, cache_bytes=128 << 20)
    lasagna_arrays = {}
    if args.lasagna:
        reference_root = Path('output/sheet_normals_20261002')
        manifest = json.loads((reference_root/'paris4/lasagna.json').read_text())
        base_url = (BUCKET+MANIFESTS['paris4']).rsplit('/', 1)[0]
        for channel in ('nx', 'ny', 'grad_mag'):
            group = manifest['groups'][channel]
            array_root, level = group['zarr'].rsplit('/', 1)
            lasagna_arrays[channel] = open_remote(base_url+'/'+array_root, int(level), reference_root/'cache')
        assert len({tuple(a.shape) for a in lasagna_arrays.values()}) == 1
        assert len({manifest['groups'][k]['scaledown'] for k in lasagna_arrays}) == 1
    root = Path('/mnt/raid_nvme/spiral_dataset_working/verified_patches')
    rng = np.random.default_rng(7431)
    rows = []
    for mesh_index, name in enumerate(('0000_low_band_final', '0000_mid_band_final', '0000_top_band')):
        site_count = args.locations // 3 + int(mesh_index < args.locations % 3)
        mesh = read_tifxyz(root/name)
        mesh.resolution = 'stored'
        x, y, z, valid = mesh[:, :]
        xyz = np.stack((x, y, z), axis=-1)
        valid = binary_erosion(valid & np.isfinite(xyz).all(-1), structure=np.ones((3, 3)))
        indices = np.argwhere(valid)
        chosen = []
        for idx in rng.permutation(len(indices)):
            r, c = indices[idx]
            pos_ct = xyz[r, c].astype(np.float64)
            if any(np.linalg.norm(pos_ct-p) < 128 for p in chosen):
                continue
            local = xyz[r-1:r+2, c-1:c+2].reshape(-1, 3).astype(np.float64)
            if np.max(np.linalg.norm(local-pos_ct, axis=1)) > 100:
                continue
            values, vectors = np.linalg.eigh(np.cov(local.T))
            if values[1] < 1 or values[0]/values[1] > .1:
                continue
            mesh_normal = vectors[:, 0]
            fit_rms = float(np.sqrt(np.mean(((local-local.mean(0)) @ mesh_normal)**2)))
            chosen.append(pos_ct)
            if args.lasagna:
                lasagna_normal, support, coherence = normal_at(lasagna_arrays, manifest, source['ct_grid_scale'], pos_ct)
            pos = pos_ct/vol.input_scale
            for family in ('H', 'V'):
                prior = ct_seed_heading(vol, pos, family)
                input_frame = prior_frames([prior])[0]
                patch, path, tensor_n, _ = model_inputs(vol, model.cfg, [pos], [input_frame], [pos[None]], normal_targets=True)
                assert not path.any()
                with torch.no_grad():
                    output = model.forward_outputs(patch, path, family_ids([family]))
                heading = input_frame @ output['heading'][0].numpy()
                if heading @ prior < 0:
                    heading = -heading
                normal = input_frame @ output['normal'][0].numpy()
                frame = orthonormal_frame(torch.from_numpy(heading), torch.from_numpy(normal)).numpy()
                tensor_normal = input_frame @ tensor_n[0].numpy()
                tensor_frame = orthonormal_frame(torch.from_numpy(heading), torch.from_numpy(tensor_normal)).numpy()
                mesh_u = mesh_normal-(mesh_normal @ heading)*heading
                mesh_u /= np.linalg.norm(mesh_u)
                row = dict(mesh=name, row=int(r), col=int(c), family=family, position_ct_xyz=pos_ct.tolist(),
                    plane_fit_rms_ct=fit_rms, mesh_normal=mesh_normal.tolist(), predicted_normal=normal.tolist(),
                    predicted_heading=heading.tolist(), tensor_normal=tensor_normal.tolist(),
                    normal_error_deg=float(angle(normal, mesh_normal)), roll_error_deg=float(angle(frame[:, 0], mesh_u)),
                    tensor_error_deg=float(angle(tensor_normal, mesh_normal)),
                    tensor_roll_error_deg=float(angle(tensor_frame[:, 0], mesh_u)),
                    heading_out_of_plane_deg=float(np.degrees(np.arcsin(np.clip(abs(heading @ mesh_normal), 0, 1)))))
                if args.lasagna:
                    lasagna_u = lasagna_normal-(lasagna_normal @ heading)*heading
                    lasagna_u /= np.linalg.norm(lasagna_u)
                    row.update(lasagna_normal=lasagna_normal.tolist(), lasagna_support=support, lasagna_coherence=coherence,
                        mesh_lasagna_deg=float(angle(mesh_normal, lasagna_normal)),
                        normal_lasagna_deg=float(angle(normal, lasagna_normal)),
                        roll_lasagna_deg=float(angle(frame[:, 0], lasagna_u)),
                        tensor_lasagna_deg=float(angle(tensor_normal, lasagna_normal)),
                        tensor_roll_lasagna_deg=float(angle(tensor_frame[:, 0], lasagna_u)))
                rows.append(row)
                print(name, r, c, family, 'normal/roll/tensor',*[round(row[k], 2) for k in
                      ('normal_error_deg', 'roll_error_deg', 'tensor_error_deg')], flush=True)
            if len(chosen) == site_count:
                break
        if len(chosen) != site_count:
            raise ValueError(f'Insufficient valid mesh locations: {name}')
    report = dict(checkpoint=str(checkpoint), step=ckpt['step'], seed=7431, locations=args.locations, predictions=len(rows),
                  protocol='Geometry-selected interior points distributed evenly across three meshes; local 3x3 plane-fit reference; '
                  'stored mesh XYZ are selected-CT voxels (metadata area ratio implies 9.6 um); '
                  'CT-initialized H/V priors, no path history, CPU inference; unsigned normal and roll angles; '
                  'roll compares normals projected around the same predicted heading; no mesh reference supplied to model; '
                  'small diagnostic, not a held-out fiber-tracing accuracy test',
                  summary={k: stats([r[k] for r in rows]) for k in
                           ('normal_error_deg', 'roll_error_deg', 'tensor_error_deg', 'tensor_roll_error_deg',
                            'heading_out_of_plane_deg')}, rows=rows)
    if args.lasagna:
        report['protocol'] += '; Lasagna unsigned outer-product interpolation with valid grad_mag support >=0.99'
        report['summary'].update({k: stats([r[k] for r in rows]) for k in
            ('mesh_lasagna_deg', 'normal_lasagna_deg', 'roll_lasagna_deg', 'tensor_lasagna_deg', 'tensor_roll_lasagna_deg')})
        report['lasagna_manifest'] = manifest
        report['lasagna_manifest_url'] = BUCKET+MANIFESTS['paris4']
    suffix = '' if args.locations == 6 else f'_sites{args.locations}'
    if args.lasagna:
        suffix += '_lasagna'
    out = Path(f'output/heading_frame_mesh_check_step{ckpt["step"]:06d}{suffix}.json')
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']), '\nSaved', out)


if __name__ == '__main__':
    main()
