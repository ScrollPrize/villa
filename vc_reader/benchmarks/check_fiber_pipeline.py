"""Check real CT crop geometry, normalization and augmentation before timing."""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vc-site-packages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.append(str(args.vc_site_packages))
    importlib.import_module('_editable_skbc_vc_reader')
    from fiber_cuda_pipeline import CudaInputs, reader_spec
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, CTVolumeSpec
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import scalar_crops
    from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, frame_from_heading
    from vesuvius.neural_tracing.tracers.fibers.data.observations import augment_ct
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    crop = CropSpec(**checkpoint['crop'])
    pipeline = CudaInputs()
    report = []
    for source in checkpoint['dataset_config']['sources']:
        doc = json.loads((args.checkpoint.parent / f'validation_{source["name"]}.json').read_text())
        spec = CTVolumeSpec.from_dict(doc['volume'])
        spec.cache_dir = checkpoint['dataset_config']['cache_dir']
        key = source['ct'] + '::' + str(source['ct_level'])
        spec.ct_normalization = checkpoint['ct_normalization']['volumes'][key]
        volume = CTVolume(spec, cache_bytes=256 << 20)
        items = [dict(pos=np.array(seed['pos']), frame=frame_from_heading(seed['heading']))
                 for seed in doc['monitor'][:2]]
        reference = scalar_crops(items, volume, crop).numpy()
        reader = dict(reader_spec(volume.ct), offline=True, cache_writes=False)
        for blur in (0., .9):
            params = (1.2, -.03, 0.)
            batch = dict(x={}, crop_pos=torch.from_numpy(np.stack([i['pos'] for i in items])),
                         crop_frame=torch.from_numpy(np.stack([i['frame'] for i in items])),
                         _vc_crop=dict(reader=reader, scale=volume.input_scale, crop=crop,
                                       photometric=[params]*len(items), blur=[blur]*len(items),
                                       seeds=list(range(len(items)))))
            if not report:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    with torch.profiler.record_function('vc_reader_input_pipeline'):
                        pipeline.materialize(batch)
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
                args.output.with_suffix('.profile.txt').write_text(
                    profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=30))
            else:
                pipeline.materialize(batch)
            actual = batch['x']['fine'].cpu().numpy()
            expected = reference.copy()
            for row in expected:
                augment_ct(row[0], params, np.random.default_rng(0), blur)
            error = np.abs(actual - expected)
            result = dict(source=source['name'], blur=blur, max_abs=float(error.max()),
                          mean_abs=float(error.mean()), reference_std=float(expected.std()))
            print(result, flush=True)
            if result['max_abs'] > 5e-4 or result['mean_abs'] > 2e-5:
                raise AssertionError(f'Crop/augmentation mismatch: {result}')
            report.append(result)
        # Noise is distribution-equivalent, with a distinct native RNG. Check
        # its scale and mean against the same noise-free transformed CT.
        noisy_params = (1.2, -.03, .02)
        batch['_vc_crop'] = dict(reader=reader, scale=volume.input_scale, crop=crop,
                                 photometric=[noisy_params]*len(items), blur=[.9]*len(items),
                                 seeds=list(range(len(items))))
        pipeline.materialize(batch)
        residual = batch['x']['fine'].cpu().numpy() - expected
        noise_result = dict(source=source['name'], noise_mean=float(residual.mean()),
                            noise_std=float(residual.std()), expected_noise_std=.16)
        if abs(noise_result['noise_mean']) > .001 or abs(noise_result['noise_std'] - .16) > .001:
            raise AssertionError(f'Noise distribution mismatch: {noise_result}')
        report.append(noise_result)
        print(noise_result, flush=True)
        # Both consumers must resolve to the same resident chunk cache. A second
        # tensor query at the same points must reuse its previously uploaded bricks.
        from vc_reader.sheet_normal import sample_sheet_normals
        gpu = pipeline.pool.get(reader, 'heading')
        assert gpu is pipeline.pool.get(reader, 'images')
        points = np.stack([i['pos'] for i in items]) * volume.input_scale
        sample_sheet_normals(gpu, points)
        before = gpu.stats
        sample_sheet_normals(gpu, points)
        after = gpu.stats
        assert after['chunk_uploads'] == before['chunk_uploads']
        assert after['cache_hits'] > before['cache_hits']
    pipeline.stats()  # rejects a CPU sampling fallback
    trace = json.loads(args.output.with_suffix('.trace.json').read_text())
    downloads = [e for e in trace['traceEvents']
                 if e.get('cat') == 'gpu_memcpy' and 'DtoH' in e.get('name', '')]
    assert all(e.get('args', {}).get('bytes', 0) <= 4 for e in downloads), downloads
    print('Image-path downloads: validation flags only:',
          sum(e.get('args', {}).get('bytes', 0) for e in downloads), 'bytes', flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
