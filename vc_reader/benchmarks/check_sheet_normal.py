"""Compare CPU/CUDA sheet normals with the independent reference on real CT."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--vc-site-packages',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    sys.path.append(str(args.vc_site_packages))
    importlib.import_module('_editable_skbc_vc_reader')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from test_sheet_normal import reference
    from vc_reader.reader import ZarrReader
    from vc_reader.sheet_normal import sheet_normals, sheet_normal_regions, sample_sheet_normals, RECIPE
    ckpt = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    rows = []
    for source in ckpt['dataset_config']['sources']:
        doc = json.loads((args.checkpoint.parent/f'validation_{source["name"]}.json').read_text())
        points = np.asarray([s['pos'] for s in doc['monitor'][:4]])*(source['grid_scale']/source['ct_grid_scale'])
        reader = ZarrReader.open(source['ct'],level=source['ct_level'],cache_dir=ckpt['dataset_config']['cache_dir'],
                                 offline=True,cache_writes=False,memory_cache=256<<20,threads=4)
        starts,offsets,valid = sheet_normal_regions(points,reader.shape)
        patches = np.stack([reader.read_region(s,(109,109,109)) for s in starts])
        before = time.perf_counter()
        cpu = sheet_normals(patches,offsets,valid=valid)
        cpu_seconds = time.perf_counter()-before
        expected = np.stack([reference(a,o) if v else np.zeros(4) for a,o,v in zip(patches,offsets,valid)])
        gpu_reader = reader.cuda(cache_bytes=256<<20)
        if not rows:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                output = sample_sheet_normals(gpu_reader,points)
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
        before = time.perf_counter()
        output = sample_sheet_normals(gpu_reader,points)
        cuda_seconds = time.perf_counter()-before
        gpu = output.numpy()  # Explicit validation-only result download, outside profile.
        for name,actual in [('cpu',cpu),('cuda',gpu)]:
            np.testing.assert_allclose(actual[:,3],expected[:,3],atol=3e-5)
            good = expected[:,3]>0
            # Angular error computed in double to measure float32 unit-vector accuracy.
            a=actual[good,:3].astype(np.float64); b=expected[good,:3]
            cos=np.abs((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)))
            angle=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
            if len(angle) and angle.max()>.01:
                raise AssertionError(f'{name} angular mismatch: {angle}')
            np.testing.assert_array_equal(actual[~good],0)
        row=dict(source=source['name'],queries=len(points),cpu_compute_seconds=cpu_seconds,
                 cuda_read_compute_seconds=cuda_seconds, max_cuda_reference_angle_degrees=float(angle.max(initial=0)),
                 max_cpu_cuda_abs=float(np.abs(cpu-gpu).max(initial=0)),reader_stats=gpu_reader.stats)
        assert row['reader_stats']['cpu_routes']==0
        rows.append(row); print(row,flush=True)
    trace=json.loads(args.output.with_suffix('.trace.json').read_text())
    transfers=[dict(name=e['name'],bytes=e.get('args',{}).get('bytes',0)) for e in trace['traceEvents'] if e.get('cat')=='gpu_memcpy']
    downloads=[e for e in transfers if 'DtoH' in e['name']]
    assert all(e['bytes']<=4 for e in downloads),downloads
    args.output.write_text(json.dumps(dict(recipe=RECIPE,rows=rows,transfers=transfers,
        transfer_note='No image/normal download inside estimator profile; explicit validation result downloads outside.'),indent=2)+'\n')


if __name__=='__main__':
    main()
