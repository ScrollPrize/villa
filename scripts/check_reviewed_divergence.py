"""Read-only cached-voxel checks at first divergence of changed replay cases."""
import itertools
import json
import argparse
from pathlib import Path
import numpy as np
import numcodecs
from reviewed_fiber_evaluation import ROOT
from vesuvius.neural_tracing.fiber_trace.geometry import decode_lasagna_normals_xyz


def read_region(root, lo, hi):
    arrays, valid = {}, np.ones(tuple(hi-lo), bool)
    for channel in ('presence', 'nx', 'ny'):
        path = root/f'fiber_{channel}.ome.zarr/1'
        meta = json.loads((path/'.zarray').read_text())
        assert meta['zarr_format'] == 2 and not meta['filters'] and meta['order'] == 'C'
        assert meta.get('dimension_separator','.') == '/' and meta['dtype'] == '|u1'
        assert np.all(lo >= 0) and np.all(hi <= meta['shape']), 'ROI outside volume'
        chunk = np.asarray(meta['chunks'])
        output = np.full(tuple(hi-lo), meta['fill_value'], dtype=meta['dtype'])
        codec = numcodecs.get_codec(meta['compressor'])
        for idx in itertools.product(*(range(int(l//c), int((h-1)//c)+1) for l,h,c in zip(lo,hi,chunk))):
            origin = np.asarray(idx)*chunk
            start = np.maximum(lo, origin)-lo
            end = np.minimum(hi, origin+chunk)-lo
            target = tuple(slice(int(l),int(h)) for l,h in zip(start,end))
            file = path.joinpath(*map(str,idx))
            if not file.is_file():
                valid[tuple(slice(int(l),int(h)) for l,h in zip(start,end))] = False
            else:
                decoded = np.frombuffer(codec.decode(file.read_bytes()), dtype=meta['dtype']).reshape(tuple(chunk))
                source = tuple(slice(int(l),int(h)) for l,h in zip(lo+start-origin,lo+end-origin))
                output[target] = decoded[source]
        arrays[channel] = output
    return arrays, valid


def stats(a):
    a = np.asarray(a)
    return dict(n=int(a.size), mean=float(a.mean()), p95=float(np.quantile(a,.95)), max=float(a.max())) if a.size else {}


def cache_for(folder, source):
    info = json.loads((folder/'full/inference_summary.json').read_text())
    matches = []
    for root in info['cache_roots'].values():
        # Match canonical input volume in cache provenance, not directory ordering.
        identity = json.loads((Path(root)/'cache_identity.json').read_text())
        if f'/{source}/' in identity['identity']['source']['locator']:
            matches.append(Path(root))
    assert len(matches) == 1, (source,matches)
    attrs = json.loads((matches[0]/'fiber_presence.ome.zarr/.zattrs').read_text())
    dataset = attrs['multiscales'][0]['datasets']
    assert len(dataset) == 1 and dataset[0]['path'] == '1'
    assert dataset[0]['coordinateTransformations'] == [{'type':'scale','scale':[2.,2.,2.]}]
    return matches[0]


def counts(folder):
    s = json.loads((folder/'full/summary.json').read_text())
    return {(v['source_id'],v['backend_config']['fibers'][c['source_index']],c['direction']):c['restart_count']
            for v in s['sources'] for c in v['cases']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=8)
    args = parser.parse_args()
    baseline = ROOT/'halfsigma_current_raw_v3'
    original = counts(baseline)
    results = []
    for chunk in (64,32,128):
        folder = ROOT/f'profile_chunks_u31500_v1/chunk{chunk}'
        candidate = counts(folder)
        for path in sorted((folder/'full/cases').glob('*/trace.json')):
            a = json.loads(path.read_text())
            p = a['provenance']
            key = (p['source_id'],p['fiber_path'],'reverse' if p['reverse'] else 'forward')
            if candidate[key] == original[key]:
                continue
            b = json.loads((baseline/'full/cases'/path.parent.name/'trace.json').read_text())
            assert all(p[k] == b['provenance'][k] for k in ('source_id','fiber_path','reverse','trace_to_base'))
            pa = np.asarray(a['replay']['segments'][0]['trace_points_base_xyz'])
            pb = np.asarray(b['replay']['segments'][0]['trace_points_base_xyz'])
            n = min(len(pa),len(pb))
            distance = np.linalg.norm(pa[:n]-pb[:n],axis=1)
            row = dict(chunk=chunk, case=int(path.parent.name), scroll=key[0],fiber=Path(key[1]).name,
                       direction=key[2],baseline_restarts=original[key],candidate_restarts=candidate[key])
            row['first_indices']={str(t):int(np.flatnonzero(distance>t)[0]) if np.any(distance>t) else None
                                  for t in (.001,.01,1.,8.)}
            regions=[]
            for index in sorted({v for v in row['first_indices'].values() if v is not None}):
                xyz = pb[index]
                center = np.floor(xyz[::-1]/2).astype(int)
                lo,hi=center-args.radius,center+args.radius+1
                av,am=read_region(cache_for(folder,key[0]),lo,hi)
                bv,bm=read_region(cache_for(baseline,key[0]),lo,hi)
                mask=am&bm
                metrics={}
                for channel in av:
                    d=np.abs(av[channel].astype(int)-bv[channel].astype(int))[mask]
                    metrics[channel]=dict(**stats(d),changed_percent=float(100*np.mean(d!=0))) if d.size else {}
                na,_=decode_lasagna_normals_xyz(av['nx'],av['ny'])
                nb,_=decode_lasagna_normals_xyz(bv['nx'],bv['ny'])
                na, nb = na.astype(np.float64), nb.astype(np.float64)
                cosine = np.sum(na*nb,axis=-1)/(np.linalg.norm(na,axis=-1)*np.linalg.norm(nb,axis=-1))
                angle=np.degrees(np.arccos(np.clip(np.abs(cosine),0,1)))
                angle[(av['nx']==bv['nx'])&(av['ny']==bv['ny'])] = 0
                strong=mask&(av['presence']>=32)&(bv['presence']>=32)
                regions.append(dict(index=index, xyz_base=xyz.tolist(),trace_distance_base=float(distance[index]),
                    valid_voxels=int(mask.sum()),total_voxels=int(mask.size),channels=metrics,
                    axis_angle_deg_presence32=stats(angle[strong]),
                    output_boundary_distance_base=float(2*np.min(np.minimum((xyz/2)%chunk,chunk-(xyz/2)%chunk)))))
            row['regions']=regions
            results.append(row)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
