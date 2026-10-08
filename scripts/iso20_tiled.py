"""Bounded parallel trace/local-dedup pipeline; native seam assembly, no global dedup."""
import copy
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import iso20_post_overlap as overlap
from lasagna.inference_provenance import atomic_write

shared=overlap.shared
TEMPLATE=overlap.SOURCE/'beam-isotropic20.config.json'


def tiles_for_bounds(low,high,core=1024,halo=64):
    if len(low)!=3 or len(high)!=3 or core<=0 or halo<=0 or any(b<=a for a,b in zip(low,high)):
        raise ValueError('invalid tiled bounds')
    ranges=[range((b-a+core-1)//core) for a,b in zip(low,high)]
    result=[]
    for index in itertools.product(*ranges):
        start=[a+i*core for a,i in zip(low,index)]
        end=[min(b,a+core) for a,b in zip(start,high)]
        result.append(dict(index_xyz=list(index),core_min=start,core_max=end,
            halo_min=[max(0,x-halo) for x in start],
            halo_max=[x+halo for x in end]))
    return result


def trace_config(template,name,tile,threads):
    config=shared.configured(template,name,'.zarr',{'--threads':threads},extra=['--capture-step-costs','--cached-anchors-only'])
    args=config['commands'][0]['args'];i=args.index('--bbox')
    args[i+1:i+7]=list(map(str,tile['halo_min']+tile['halo_max']))
    return config


def run_tiles(tiles,worker,concurrency,on_complete):
    if concurrency<1:raise ValueError('tile concurrency must be positive')
    # Continuous completion-driven scheduling, not barriers between small batches.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending={pool.submit(worker,i,t):i for i,t in enumerate(tiles)}
        for future in as_completed(pending):on_complete(pending[future],future.result())


def main(pilot=False):
    low=[9728,22272,6400] if pilot else [9728,21504,5632]
    high=[11776,22528,6656] if pilot else [11776,23552,7680]
    out=overlap.OUT.parent/('20261004_iso20_tiled_pilot_v4' if pilot else '20261004_iso20_tiled2048_v2')
    out.mkdir(parents=True,exist_ok=True)
    shared.OUT=out;overlap.OUT=out;overlap.LOW=False
    template=json.loads(TEMPLATE.read_text())
    tiles=tiles_for_bounds(low,high)
    workers=min(4,len(tiles));threads=128//workers
    identity=dict(low=low,high=high,core=1024,halo=64,workers=workers,threads=threads)
    shared.save_config(out/'tiled.config.json',identity)
    started=time.monotonic();records={}
    print(f'Tiled tracing: {len(tiles)} tiles, {workers} concurrent, {threads} native threads each',flush=True)
    # One shared writer before read-only parallel tile processes. No duplicate
    # anchor caches and no cross-process exclusive-lock contention.
    preparation=shared.configured(template,'prepare-anchors','.json',{'--threads':128},extra=['--prepare-anchors-only'])
    args=preparation['commands'][0]['args'];i=args.index('--bbox')
    expanded_low=[min(t['halo_min'][a] for t in tiles) for a in range(3)]
    expanded_high=[max(t['halo_max'][a] for t in tiles) for a in range(3)]
    args[i+1:i+7]=list(map(str,expanded_low+expanded_high))
    prepared=shared.execute(preparation)

    def worker(index,tile):
        name=f'tile{index:03d}'
        trace=shared.execute(trace_config(template,name,tile,threads))
        cfg=overlap.filter_config(template,name+'-dedup','presence',10,20,0,source_trace=out/(name+'.zarr'))
        args=cfg['commands'][0]['args'];args[args.index('--threads')+1]=str(threads)
        filtered=shared.execute(cfg)
        artifact=out/(name+'-dedup.zarr')
        attrs=json.loads((artifact/'.zattrs').read_text())['processing']['artifact']
        return dict(**tile,artifact=str(artifact),trace_seconds=trace['wall_seconds'],
                    dedup_seconds=filtered['wall_seconds'],retained=attrs['trace_count'])

    def completed(i,row):
        records[i]=row
        atomic_write(out/'progress.json',dict(completed=len(records),total=len(tiles),
            elapsed=time.monotonic()-started,tiles=[records[k] for k in sorted(records)]),durable=False)
        print(f'Tile {i} complete ({len(records)}/{len(tiles)}): {row["retained"]} retained; '
              f'trace {row["trace_seconds"]:.1f}s, dedup {row["dedup_seconds"]:.1f}s',flush=True)

    run_tiles(tiles,worker,workers,completed)
    manifest=dict(contract=1,minimum=low,maximum=high,
        tiles=[{k:r[k] for k in ('index_xyz','core_min','core_max','halo_min','halo_max','artifact')}
               for _,r in sorted(records.items())],
        seam=dict(normal_radius=10,tangential_radius=20,angle_degrees=45,sample_step=4,
                  minimum_support=32,minimum_fraction=.8))
    shared.save_config(out/'tiles.json',manifest)
    config=copy.deepcopy(template);config['output_root']=str(out)
    config['commands']=[dict(name='assembled',binary='vc_fiber_trace_chunk',args=[
        'assemble-tiles',str(out/'tiles.json'),'--output',str(out/'assembled.zarr'),
        '--normal-manifest',template['commands'][0]['args'][template['commands'][0]['args'].index('--normal-manifest')+1],
        '--threads','128'])]
    config['input_files']=[*template['input_files'],str(out/'tiles.json'),
                          *[str(Path(r['artifact'])/'.zattrs') for _,r in sorted(records.items())]]
    record=shared.execute(config)
    attrs=json.loads((out/'assembled.zarr/.zattrs').read_text())
    topology=attrs['processing']['preprocessing']['tiled_trace']
    ambiguous=sum(not c['unique'] for c in topology['candidates'])
    result=dict(**identity,tiles=[records[k] for k in sorted(records)],
        anchor_preparation_seconds=prepared['wall_seconds'],assembly_seconds=record['wall_seconds'],wall_seconds=time.monotonic()-started,
        fragments=len(topology['fragments']),seam_candidates=len(topology['candidates']),
        ambiguous_candidates=ambiguous,unique_candidates=len(topology['candidates'])-ambiguous,
        endpoint_counts=topology['endpoint_counts'],
        bp_status='not run: unresolved alternatives or unmatched internal cuts' if ambiguous or topology['endpoint_counts']['unmatched_internal'] else 'unique seams supported; BP not yet run')
    atomic_write(out/'results.json',result,durable=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    import sys
    main('--pilot' in sys.argv)
