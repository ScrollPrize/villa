"""Bounded same-evidence comparison of Adam and wrapped-IRLS stage backends.

The explicit external inputs are an AFV source/cohort and archived sign/HV
tensor caches. Only identity conversion and induced-subgraph selection occur;
neither measured evidence nor existing H/V stage exclusions are re-estimated.
Call under the normal managed GPU launcher, not as an unregistered CUDA job.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import torch
from vc_object_store import AfvSource, CollectionReader
from lasagna.inference_provenance import atomic_write, sha256_file

from .fiber_graph_store import store_digest
from . import fiber_hv_stage, fiber_normal_sign


def prepare_cached(config):
    """Use immutable stage tensors as-is; benchmark outputs retain their IDs."""
    started=time.monotonic()
    with np.load(config['normal_cache']) as normal,np.load(config['hv_cache']) as hv,np.load(config['length_catalog']) as catalog:
        ids=normal['ids']
        if not np.array_equal(ids,hv['ids']) or not np.array_equal(ids,catalog['ids']):
            raise ValueError('Cached stage/catalog identities differ')
        lengths=catalog['lengths']
        if lengths.shape!=ids.shape or not np.isfinite(lengths).all() or np.any(lengths<=0):
            raise ValueError('Invalid whole-fiber lengths')
        if not np.array_equal(hv['active'],np.flatnonzero(~hv['inherited'])):
            raise ValueError('Invalid inherited exclusions')
        np.testing.assert_allclose(lengths[hv['active']],hv['lengths'],rtol=1e-6,atol=1e-5)
        report=dict(import_wall_seconds=0.,fibers=len(ids),normal_events=len(normal['i']),
            hv_events=len(hv['i']),inherited_broken=int(hv['inherited'].sum()),
            total_length_base=float(lengths.sum()),
            cohort='complete cached graph; archived H/V-stage exclusions fixed; no extraction or import',
            preparation_seconds=time.monotonic()-started)
    Path(config['output_root']).mkdir(parents=True,exist_ok=False)
    atomic_write(Path(config['output_root'])/'preparation.json',report,durable=False)
    return report


def prepare(config):
    root=Path(config['output_root'])
    root.mkdir(parents=True,exist_ok=False)
    store=root/'fibers.vcobj'
    native=dict(source=config['source'],output=str(store),volume_id=config['volume_id'],
        scale_factor=1.,shards=config.get('workers',8),readers=config.get('workers',8),
        packed=True,memory_first=True,report=str(root/'import_report.json'),verify_readback=False)
    atomic_write(root/'import_config.json',native,durable=False)
    started=time.monotonic()
    subprocess.run([config['native_import_binary'],str(root/'import_config.json')],check=True,timeout=30)
    import_seconds=time.monotonic()-started
    with np.load(config['selected_cache']) as selected:
        source_ids=selected['source_id'];lengths=selected['lengths']
    if len(np.unique(source_ids))!=len(source_ids):
        raise ValueError('Trial requires whole unique source fibers, not split pieces')
    namespace=uuid.UUID(AfvSource(config['source']).uuid)
    ids=np.asarray([str(uuid.uuid5(namespace,'fiber:'+str(s))) for s in source_ids])
    order=np.argsort(ids);ids=ids[order];source_ids=source_ids[order];lengths=lengths[order]
    with CollectionReader(store) as reader:
        if set(ids)!=set(r['uuid'] for r in reader.objects('fiber')):
            raise ValueError('Imported fibers differ from the requested complete cohort')
    signature=dict(database_sha256=store_digest(store))
    started=time.monotonic()
    with np.load(config['normal_cache']) as d:
        all_ids=d['ids'];global_indices=np.searchsorted(all_ids,source_ids)
        if not np.array_equal(all_ids[global_indices],source_ids):
            raise ValueError('Cohort missing from normal evidence')
        remap=np.full(len(all_ids),-1,np.int64);remap[global_indices]=np.arange(len(ids))
        i,j=remap[d['i']],remap[d['j']];valid=(i>=0)&(j>=0)
        np.savez(root/'normal_tensors.npz',ids=ids,prior=d['prior'][global_indices],
                 i=i[valid],j=j[valid],dot=d['dot'][valid],lengths=lengths)
        sign_events=int(valid.sum())
    atomic_write(root/'normal_tensors.json',dict(signature=signature,fibers=len(ids),used_events=sign_events),durable=False)
    with np.load(config['hv_cache']) as d:
        if not np.array_equal(d['ids'],all_ids):
            raise ValueError('H/V and normal source catalogs differ')
        inherited=d['inherited'][global_indices]
        active=np.flatnonzero(~inherited)
        active_in_cache=np.searchsorted(d['active'],global_indices[active])
        if not np.array_equal(d['active'][active_in_cache],global_indices[active]):
            raise ValueError('H/V active catalog mismatch')
        remap=np.full(len(d['active']),-1,np.int64);remap[active_in_cache]=np.arange(len(active))
        i,j=remap[d['i']],remap[d['j']];valid=(i>=0)&(j>=0)
        i,j=i[valid],j[valid];relation=d['relation'][valid]
        degree=np.bincount(np.r_[i,j],minlength=len(active)).astype(np.float32)
        np.savez(root/'hv_tensors.npz',ids=ids,active=active,inherited=inherited,
            a=d['a'][active_in_cache],lengths=lengths[active],i=i,j=j,relation=relation,degree=degree)
        hv_events=len(i)
    stats=dict(signature=signature,all_fibers=len(ids),active_fibers=len(active),inherited_broken=int(inherited.sum()),
               original_parallel=int((relation>0).sum()),original_perpendicular=int((relation<0).sum()))
    atomic_write(root/'hv_tensors.json',stats,durable=False)
    report=dict(import_wall_seconds=import_seconds,preparation_seconds=time.monotonic()-started,
                fibers=len(ids),normal_events=sign_events,hv_events=hv_events,inherited_broken=int(inherited.sum()),
                total_length_base=float(lengths.sum()),cohort='induced whole-fiber subgraph; archived H/V-stage exclusions fixed in all trials',
                import_report=json.loads((root/'import_report.json').read_text()))
    atomic_write(root/'preparation.json',report,durable=False)
    return report


def run(config):
    if not torch.cuda.is_available():
        raise RuntimeError('Managed CUDA trial required')
    root=Path(config['output_root'])
    cache_only=bool(config.get('cache_only',False))
    preparation=prepare_cached(config) if cache_only else prepare(config)
    rows=[]
    for stage in ('normal_sign','classification'):
        for backend,entropy in [('adam',0.),('wrapped_irls',0.)]+([('adam',.01)] if stage=='classification' else []):
            name=f'{stage}_{backend}_entropy{entropy:g}'
            output=root/name
            cache=(config['normal_cache' if stage=='normal_sign' else 'hv_cache'] if cache_only else
                   str(root/('normal_tensors.npz' if stage=='normal_sign' else 'hv_tensors.npz')))
            cfg=dict(config.get('parameters',{}),database=str(root/'fibers.vcobj'),tensor_cache=cache,
                output_root=str(output),optimizer=backend,entropy_weights=[entropy],write_result_store=not cache_only)
            atomic_write(root/(name+'_config.json'),cfg,durable=False)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            started=time.monotonic()
            module=fiber_normal_sign if stage=='normal_sign' else fiber_hv_stage
            module.solve(cfg)
            torch.cuda.synchronize()
            summary=json.loads((output/'summary.json').read_text())
            elapsed=time.monotonic()-started
            for item in summary['runs']:
                final=item['final']
                row=dict(stage=stage,backend=backend,entropy=entropy,wall_seconds=elapsed,
                    solve_seconds=final['seconds'],iterations=final['step'],
                    peak_allocated_bytes=summary['peak_allocated_bytes'],metrics=final)
                path=output/(f"solution_{item['broken_penalty']:g}.npz" if stage=='normal_sign' else f"solution_{item['name']}.npz")
                # Use saved hard outputs for retention, including historical Adam.
                if stage=='normal_sign':
                    with np.load(path) as s,np.load(config['length_catalog'] if cache_only else root/'normal_tensors.npz') as d:
                        row['retained_arc_percent']=float(100*d['lengths'][~s['broken']].sum()/d['lengths'].sum())
                else:
                    row['retained_arc_percent']=final['retained_arc_percent']
                row['result_arrays']=str(path)
                row['result_store']=None if cache_only else str(path.with_suffix('.vcobj'))
                if not cache_only:
                    with CollectionReader(path.with_suffix('.vcobj')) as reader:
                        saved=reader.extension_rows('ext_fiber_graph_solve_fibers')
                        if len(saved)!=preparation['fibers']:
                            raise AssertionError('Typed result is missing source fibers')
                rows.append(row)
            report=dict(config=config,preparation=preparation,results=rows,
                gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                code_sha256={name:sha256_file(Path(__file__).with_name(name+'.py')) for name in
                    ('fiber_wrapped_irls','fiber_wrapped_stages','fiber_wrapped_benchmark','fiber_normal_sign','fiber_hv_stage')},
                timing='One repetition, nonexclusive manager grant. solve_seconds includes periodic diagnostics; wall_seconds also includes loading and output persistence. cache_only runs save arrays, not typed stores.')
            atomic_write(root/'comparison.json',report,durable=False)
            print('TRIAL COMPLETE '+json.dumps(rows[-1]),flush=True)
    return report


if __name__=='__main__':
    import argparse
    import sys
    from .managed_evaluation import launch_managed
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--devices')
    parser.add_argument('--unmanaged',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.unmanaged:
        run(cfg)
    else:
        command=[sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_wrapped_benchmark','--config',str(args.config.resolve())]
        raise SystemExit(launch_managed(command,devices=args.devices or cfg.get('devices','0'),settings=cfg['gpu_manager']))
