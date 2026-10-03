"""Inspect paired follower crops with a fixed model heading and tensor versus learned roll.

OMP_NUM_THREADS=1 python evaluation/visualize_future_frames.py
Selects median-disagreement and maximum-disagreement seed states for H/V in each volume.
Uses saved predictions from compare_future_crop_frames.py; no new inference or annotation-derived rotations.
"""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config, ct_source_spec
from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, default=Path('output/future_crop_frames_step064000/report.json'))
    ap.add_argument('--out', type=Path, default=Path('output/future_crop_frames_step064000/visuals'))
    args = ap.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fiber_follow_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text())
    crop = CropSpec(**report['crop'])
    document, _ = read_dataset_config('configs/mixed_ct_datasets_paris50.json')
    normalization = json.loads((args.report.parent/'ct_normalization.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in document['sources']:
        name = source['name']
        spec = ct_source_spec(source, document['cache_dir'])
        spec.ct_normalization = normalization['volumes'][volume_key(spec)]
        vol = FiberVolume(spec, cache_bytes=128 << 20, cache_only=True)
        selected = []
        for family in 'HV':
            candidates = [r for r in report['rows'] if r['source']==name and r['family']==family and r['history']==0]
            def disagreement(row):
                a, b = [np.array(row['frames'][k])[:, 0] for k in ('model_frame', 'model_tensor_roll')]
                return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))
            candidates.sort(key=disagreement)
            for category, row in [('typical', candidates[len(candidates)//2]), ('largest disagreement', candidates[-1])]:
                selected.append((category, row, disagreement(row)))
        figure, axes = plt.subplots(4, 6, figsize=(21, 14), constrained_layout=True)
        for i, (category, row, delta) in enumerate(selected):
            frames = np.stack([row['frames'][k] for k in ('model_tensor_roll', 'model_frame')])
            # Unsigned normals have equivalent u/v flips; align signs for a fair visual comparison.
            if frames[0, :, 0] @ frames[1, :, 0] < 0:
                frames[1, :, :2] *= -1
            cubes = scalar_crops([dict(pos=np.array(row['position']), frame=f) for f in frames],
                                 vol, crop, normalize=False).numpy()[:, 0]
            vmax = float(np.percentile(cubes, 99.5))
            vmin = 0.
            ident = f'{name}_{row["family"]}_{row["index"]}'
            np.savez_compressed(args.out/(ident+'.npz'), cubes=cubes, frames=frames,
                                position=row['position'], future=row['future'], crop_json=json.dumps(report['crop']))
            detail, detail_axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
            lateral = crop.lateral_coords
            forward = crop.forward_coords
            # Even cross-section dimensions: average the two central planes to evaluate the head-centered plane.
            center = slice(crop.width//2-1, crop.width//2+1)
            for j, (cube, frame) in enumerate(zip(cubes, frames)):
                views = [cube[:, :, center].mean(-1).T, cube[:, center, :].mean(1).T, cube[crop.behind]]
                extents = [[forward[0], forward[-1], lateral[0], lateral[-1]]]*2 + [[lateral[0], lateral[-1]]*2]
                local = np.asarray(row['future']) @ frame
                for k, (view, extent) in enumerate(zip(views, extents)):
                    for ax in (axes[i, j*3+k], detail_axes[j, k]):
                        ax.imshow(view, cmap='gray', origin='lower', extent=extent, vmin=vmin, vmax=vmax,
                                  interpolation='nearest', aspect='equal')
                        ax.plot(0, 0, '+', color='#ffb000', markersize=7, markeredgewidth=1)
                        if k < 2:
                            vertical, out_of_plane = ((1, 0) if k==0 else (0, 1))
                            # Show only annotation samples within one CT voxel of this central plane.
                            visible = abs(local[:, out_of_plane]) <= crop.spacing
                            ax.scatter(local[visible, 2], local[visible, vertical], s=4, color='#00e5ff', alpha=.75)
                        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.tick_params(labelsize=6)
                    detail_axes[j, k].set_title(('2/8 tensor roll' if j==0 else 'Learned roll')+' | '+
                                               ('Face-on (f,v)', 'Side (f,u)', 'End-on (u,v)')[k], fontsize=10)
                    detail_axes[j, k].set_xlabel('Forward' if k<2 else 'Sheet normal')
                    detail_axes[j, k].set_ylabel(('In-sheet transverse', 'Sheet normal', 'In-sheet transverse')[k])
                axes[i, 0].set_ylabel(f'{row["family"]} #{row["index"]}\n{category}\nroll difference {delta:.1f}°', fontsize=10)
            detail.suptitle(f'{name} | {row["family"]} #{row["index"]} | {category} | Δroll {delta:.1f}°\n'
                           'Same heading and CT contrast; + seed; cyan: future annotation within the central plane', fontsize=12)
            detail.savefig(args.out/(ident+'.png'), dpi=170)
            plt.close(detail)
            manifest.append(dict(source=name, index=row['index'], family=row['family'], category=category,
                roll_difference_deg=delta, position=row['position'], image=ident+'.png', data=ident+'.npz',
                display_range=[vmin, vmax]))
            print(name, row['family'], row['index'], category, round(delta, 2), flush=True)
        for j, title in enumerate(('Tensor face-on', 'Tensor side', 'Tensor end-on',
                                   'Model face-on', 'Model side', 'Model end-on')):
            axes[0, j].set_title(title, fontsize=12)
        figure.suptitle(f'{name}: same model heading, different roll | central CT planes | step {report["step"]}', fontsize=16)
        figure.savefig(args.out/(name+'_overview.png'), dpi=140)
        plt.close(figure)
    (args.out/'manifest.json').write_text(json.dumps(dict(step=report['step'], crop=report['crop'], sites=manifest), indent=2)+'\n')
    body = ''.join(f'<h2>{html.escape(m["source"])} {m["family"]} #{m["index"]} — {m["category"]}</h2>'
                   f'<a href="{m["image"]}"><img src="{m["image"]}" style="width:100%;max-width:1400px"></a>' for m in manifest)
    (args.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Crop orientation comparison</title>'
        '<body style="font-family:sans-serif;background:#eee;margin:2em"><h1>Tensor versus learned roll</h1>'
        '<p>Same 64k model heading; actual follower CT crop. Median and largest roll disagreement among H/V seed states '
        'in each volume. Tensor and model share CT contrast and normal sign. Cross-section axes are centered between '
        'the two middle samples. Gold cross: seed. Cyan: annotated future points within one CT voxel of the plane.</p>'+body)


if __name__ == '__main__':
    main()
