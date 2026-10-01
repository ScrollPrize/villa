"""Compare final cache bytes on common chunk coordinates; never compare fill values."""
import itertools
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import numcodecs

NAMES = [f'{d}_r{r}' for r in (1,2,3) for d in ('float16','float32')]
PAIRS = list(itertools.combinations(range(6),2))
CHANNELS = ('presence','nx','ny')


def bounded_map(pool, function, items):
    items=iter(items)
    pending=deque(pool.submit(function,x) for x in itertools.islice(items,32))
    while pending:
        yield pending.popleft().result()
        for item in itertools.islice(items,1):
            pending.append(pool.submit(function,item))


def histogram_stats(hist):
    hist = np.asarray(hist,dtype=np.int64)
    total = int(hist.sum())
    if not total:
        return {'voxels':0}
    cdf = hist.cumsum()
    return dict(voxels=total, changed_percent=100*(total-int(hist[0]))/total,
        mae=float(np.dot(hist,np.arange(256,dtype=np.float64))/total),
        rmse=float(np.sqrt(np.dot(hist,np.arange(256,dtype=np.float64)**2)/total)),
        p95=int(np.searchsorted(cdf,np.ceil(total*.95))),
        p99=int(np.searchsorted(cdf,np.ceil(total*.99))),max=int(np.flatnonzero(hist)[-1]))


def decode_chunk(path, meta):
    value = np.frombuffer(numcodecs.get_codec(meta['compressor']).decode(path.read_bytes()),dtype=np.uint8)
    return value.reshape(meta['chunks'])


def inspect_cache(root):
    result = {}
    for channel in CHANNELS:
        path = root/f'fiber_{channel}.ome.zarr/1'
        meta = json.loads((path/'.zarray').read_text())
        assert meta['dtype'] == '|u1' and meta['order'] == 'C' and not meta['filters']
        assert meta['dimension_separator'] == '/' and meta['chunks'] == [64,64,64]
        keys = {tuple(map(int,(p.parent.parent.name,p.parent.name,p.name)))
                for p in path.glob('*/*/*') if p.is_file() and
                all(x.isdecimal() for x in (p.parent.parent.name,p.parent.name,p.name))}
        result[channel] = (path,meta,keys)
    return result


def compare_source(caches):
    result = {}
    for channel in CHANNELS:
        entries = [c[channel] for c in caches]
        assert all(e[1]['shape'] == entries[0][1]['shape'] for e in entries)
        shape = np.array(entries[0][1]['shape'])
        union = set.union(*(e[2] for e in entries))
        hist = np.zeros((len(PAIRS),256),np.int64)
        changed_chunks = np.zeros(len(PAIRS),np.int64)
        common_chunks = np.zeros(len(PAIRS),np.int64)
        union_chunks = np.zeros(len(PAIRS),np.int64)

        def one(key):
            arrays=[]
            extents=np.minimum(64,shape-np.array(key)*64)
            region=tuple(slice(0,int(s)) for s in extents)
            for path,meta,keys in entries:
                arrays.append(decode_chunk(path.joinpath(*map(str,key)),meta)[region]
                              if key in keys else None)
            local=np.zeros((len(PAIRS),256),np.int64)
            common=np.zeros(len(PAIRS),np.int64)
            either=np.zeros(len(PAIRS),np.int64)
            for k,(i,j) in enumerate(PAIRS):
                a,b=arrays[i],arrays[j]
                either[k]=a is not None or b is not None
                if a is not None and b is not None:
                    common[k]=1
                    delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
                    local[k]=np.bincount(delta.ravel(),minlength=256)
            return local,common,either

        with ThreadPoolExecutor(max_workers=16) as pool:
            for local,common,either in bounded_map(pool,one,sorted(union)):
                hist+=local
                common_chunks+=common
                union_chunks+=either
                changed_chunks+=np.any(local[:,1:]!=0,axis=1)
        result[channel]=[dict(a=NAMES[i],b=NAMES[j],common_chunks=int(common_chunks[k]),
            union_chunks=int(union_chunks[k]),changed_chunks=int(changed_chunks[k]),
            histogram=hist[k].tolist(),**histogram_stats(hist[k])) for k,(i,j) in enumerate(PAIRS)]
    return result


def report(root):
    numcodecs.blosc.set_nthreads(1)
    caches=[]
    runs=[]
    cases=[]
    source_identities={}
    model_identity=None
    protocol=None
    producer=None
    for name in NAMES:
        full=root/name/'full'
        summary=json.loads((full/'summary.json').read_text())
        assert summary['completed']
        info=json.loads((full/'inference_summary.json').read_text())
        current_protocol={k:summary[k] for k in ('effective_trace','selected_fibers',
            'requested_directed_reference_length_mm','reference_selection_policy','restart_episode_distance_base_voxels')}
        if protocol is not None:
            assert protocol == current_protocol
            assert producer == info['producer_implementation']
        protocol=current_protocol
        producer=info['producer_implementation']
        pool=info['worker_pool']
        expected_record=64**3*(18 if name.startswith('float16') else 32)
        assert pool['accumulator_record_bytes'] == expected_record
        assert pool['accumulator_hot_capacity_bytes'] == 504*1024**3
        assert not any(s['warm_chunks'] for s in info['sources'].values())
        cache={}
        for path in info['cache_roots'].values():
            p=Path(path)
            identity=json.loads((p/'cache_identity.json').read_text())
            source=identity['identity']['source']['coordinate_space'].split('/')[0]
            value=identity['identity']['source']
            if source in source_identities:
                assert source_identities[source] == value
            source_identities[source]=value
            model=dict(identity['identity']['model'])
            model.pop('accumulator_dtype')
            if model_identity is not None:
                assert model_identity == model
            model_identity=model
            assert source not in cache
            assert identity['accumulator_dtype'] == name.split('_')[0]
            cache[source]=inspect_cache(p)
        caches.append(cache)
        counts={}
        sources={}
        for source in summary['sources']:
            source_id=source['source_id']
            sources[source_id]=sum(c['restart_count'] for c in source['cases'])
            for c in source['cases']:
                key=(source_id,source['backend_config']['fibers'][c['source_index']],c['direction'])
                counts[key]=c['restart_count']
        cases.append(counts)
        runs.append(dict(name=name,total_restarts=sum(counts.values()),sources=sources,
            wall_seconds=summary['wall_seconds'],tiles=pool['tiles'],
            accumulator_record_bytes=pool['accumulator_record_bytes'],
            accumulator_hot_capacity_bytes=pool['accumulator_hot_capacity_bytes'],
            shared_slot_bytes=pool['shared_slot_bytes']))
    assert all(set(c)==set(cases[0]) for c in cases)
    assert all(set(c)==set(caches[0]) for c in caches)
    scrolls={}
    for source in sorted(caches[0]):
        print(f'Comparing cache bytes: {source}',flush=True)
        scrolls[source]=compare_source([c[source] for c in caches])
    overall={channel:[] for channel in CHANNELS}
    for channel in CHANNELS:
        for k,(i,j) in enumerate(PAIRS):
            rows=[s[channel][k] for s in scrolls.values()]
            hist=np.sum([r['histogram'] for r in rows],axis=0)
            overall[channel].append(dict(a=NAMES[i],b=NAMES[j],**histogram_stats(hist),
                common_chunks=sum(r['common_chunks'] for r in rows),
                union_chunks=sum(r['union_chunks'] for r in rows),
                changed_chunks=sum(r['changed_chunks'] for r in rows)))
    differences=[]
    for key in cases[0]:
        values=[c[key] for c in cases]
        if len(set(values))>1:
            differences.append(dict(scroll=key[0],fiber=key[1],direction=key[2],restarts=dict(zip(NAMES,values))))
    variability={}
    for dtype in ('float16','float32'):
        selected=[r for r in runs if r['name'].startswith(dtype)]
        def summarize(values):
            return dict(values=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),
                        min=min(values),max=max(values))
        variability[dtype]=dict(total=summarize([r['total_restarts'] for r in selected]),
            sources={s:summarize([r['sources'][s] for r in selected]) for s in selected[0]['sources']})
    return dict(runs=runs,variability=variability,case_differences=differences,overall=overall,scrolls=scrolls,
        note='Final quantized uint8 caches. All common voxels including zeros; missing chunks excluded. N=3 each; arrival-order policy unchanged, actual ordering uncontrolled.')
