"""Read-only real-state diagnostic equivalence and CPU timing, approved runner."""
import cProfile
import json
from pathlib import Path
import time

import numpy as np


def run(config):
    from . import fiber_uv_layout as uv
    from . import fiber_reference_comparison as ref
    from lasagna.inference_provenance import atomic_write
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    source=Path(config['diagnostic_benchmark'])
    cfg=json.loads((source/'summary.json').read_text())['config']
    with np.load(cfg['tensor_cache']) as archive:d={k:archive[k] for k in archive.files}
    with np.load(source/'solution.npz') as archive:
        state=archive['state'];widths=archive['widths']
    projection=ref.load_metric_projection(cfg['reference_projection'],d,cfg)
    cases=[('final',state,widths)]
    initial,width0=uv.fresh_initial_state(d,cfg)
    cases.insert(0,('initial',initial,width0))
    output=[]
    for label,state,widths in cases:
        geometry=uv.hard_geometry(d,state,widths,cfg)
        keep,values,weights,starts,offsets=ref.reference_metric_inputs(d,state,widths,cfg,projection,geometry)
        expected=np.array([ref.weighted_median(values[a:b],weights[a:b]) for a,b in zip(offsets[:-1],offsets[1:])])
        actual=ref.grouped_weighted_median(values,weights,offsets)
        np.testing.assert_array_equal(actual,expected)
        def evaluate(backend):
            c=dict(cfg,diagnostic_backend=backend)
            geo=uv.hard_geometry(d,state,widths,c) if backend=='optimized' else None
            return dict(audit=uv.audit(d,state,widths,c,geometry=geo),
                        reference=ref.reference_winding_metrics(d,state,widths,c,projection,geometry=geo))
        old=evaluate('reference');new=evaluate('optimized')
        if old!=new:raise AssertionError('Diagnostic dictionaries differ')
        row=dict(state=label,per_source_medians_exact=True,metrics_exact=True,groups=len(actual),timings={})
        for backend in ('reference','optimized'):
            times=[]
            for _ in range(int(config.get('repeats',3))):
                started=time.monotonic();evaluate(backend);times.append(time.monotonic()-started)
            row['timings'][backend]=dict(seconds=times,mean=float(np.mean(times)),
                p50=float(np.median(times)),p95=float(np.quantile(times,.95)))
        profile=cProfile.Profile();profile.enable();evaluate('optimized');profile.disable()
        profile.dump_stats(str(root/(label+'.prof')))
        output.append(row);atomic_write(root/'summary.json',dict(config=config,results=output),durable=False)
        print('DIAGNOSTIC BENCHMARK '+json.dumps(row),flush=True)
    return output
