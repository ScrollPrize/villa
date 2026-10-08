"""No-accumulation inference repeatability using production reader/model/finalizer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from reviewed_fiber_evaluation import ROOT, REPO
import numpy as np

OUTPUT=ROOT/'tile_repeatability_u31500_v1'


def choose_tile(geometry,xyz,source_to_base):
    center=np.array(xyz[::-1])/source_to_base
    origin=geometry.origin(tuple((center//geometry.chunk).astype(int)))
    return min(geometry.contributors(origin),
        key=lambda p:float(np.sum((np.array(p)-geometry.border+geometry.tile/2-center)**2)))


def value_stats(a,b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape mismatch or nonfinite output')
    d=np.abs(a.astype(np.float32)-b.astype(np.float32))
    return dict(values=int(d.size),changed_percent=float(100*np.count_nonzero(d)/d.size),
        mae=float(d.sum(dtype=np.float64)/d.size),
        rmse=float(np.sqrt(np.square(d,dtype=np.float64).sum()/d.size)),max=float(d.max()),nonfinite=0)


def compare(a,b,qa,qb):
    raw={str(i):value_stats(a[:,i],b[:,i]) for i in range(7)}
    rounded={k:value_stats(qa[k],qb[k]) for k in qa}
    changed=qa['presence']!=qb['presence']
    margins=np.abs(np.mod(a[:,6][changed]*255.,1.)-.5)
    rounding=dict(changed_values=int(changed.sum()),
        raw_equal_but_byte_changed=int(np.count_nonzero(changed&(a[:,6]==b[:,6]))),
        mean_raw_delta_at_changed_bytes=float(np.abs(a[:,6][changed]-b[:,6][changed]).mean()) if changed.any() else 0.,
        mean_distance_to_rounding_boundary_bytes=float(margins.mean()) if margins.size else 0.)
    return dict(raw=raw,rounded=rounded,presence_rounding=rounding)


def finalize(adapter,raw):
    product=adapter.output_products[0]
    result={c.name:np.empty((raw.shape[0],*raw.shape[2:]),np.uint8) for c in product.channels}
    jobs=[(i,z) for i in range(raw.shape[0]) for z in range(0,raw.shape[2],16)]
    def one(job):
        i,z=job
        values=adapter.finalize_product_slab(product,raw[i,:,z:z+16])
        for channel,value in values.items():
            result[channel][i,z:z+16]=value
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one,jobs))
    return result


def worker(float32_sweep=False):
    import torch
    from lasagna import tiled_predict3d as tiled
    from lasagna.demand_predict3d import DemandGeometry
    from lasagna.inference_provenance import atomic_write,sha256_file
    from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import NativeTileReader,native_reader_options
    from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
    from vesuvius.neural_tracing.fiber_trace_3d.infer import _checkpoint_config_for_inference
    limits=tiled._limit_native_worker_threads()
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    workload=json.loads((ROOT/'halfsigma_current_raw_v3/workload.json').read_text())
    assert sha256_file(workload['checkpoint']) == workload['checkpoint_sha256']
    config,_=_checkpoint_config_for_inference(workload['checkpoint'],None)
    adapter=FiberTrace3DPredictAdapter(config,checkpoint=workload['checkpoint'],level=1,scaledown=2)
    adapter.config['training']={**adapter.config.get('training',{}),'mixed_precision':'bf16'}
    model=adapter.load_model(device=device)
    observed=[]
    hook=model.register_forward_hook(lambda m,args,out:observed.append(str(out.dtype)))
    full=json.loads((ROOT/'halfsigma_current_raw_v3/full.json').read_text())
    cache_root=Path(subprocess.check_output([str(Path(workload['native_binary']).with_name('vc_cache_root'))],text=True).strip())
    sources={s['id']:s for s in full['sources']}
    # Previously examined first-divergence locations, source baseXYZ.
    probes=[('PHercParis4',[16659.91015625,8288.6904296875,34716.62109375]),
            ('PHercParis4',[10422.154296875,13791.169921875,52847.34765625]),
            ('PHerc0139',[18808.806640625,14414.2451171875,45236.375]),
            ('PHerc0139',[17109.6328125,10644.4814453125,11206.9765625])]
    readers={}
    inputs=[]
    provenance=[]
    for source,xyz in probes:
        if source not in readers:
            entry=sources[source]
            sampler=Vc3dCoordinateSampler(entry['input_volume'],level=entry['input_level'],cache_root=str(cache_root),
                **native_reader_options(full['inference']))
            geometry=DemandGeometry(tuple(sampler.volume.shape_at(sampler.level)),256,96,32,1,64)
            readers[source]=(NativeTileReader(sampler,geometry,sampler.volume.dtype),geometry)
        reader,geometry=readers[source]
        assert sources[source]['input_level']==1
        tile=choose_tile(geometry,xyz,2)
        raw=reader(tile)
        tensor=tiled._input_tile_to_device(raw,device)
        normalized=adapter.preprocess_tile(tensor,torch.ones_like(tensor,dtype=torch.bool))
        inputs.append(normalized)
        provenance.append(dict(source=source,input_level=1,source_to_base=2,probe_base_xyz=xyz,tile_zyx=tile,input_start_zyx=[v-32 for v in tile],
            input_shape=list(raw.shape),input_dtype=str(raw.dtype),input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            normalized_sha256=hashlib.sha256(normalized.cpu().numpy().tobytes()).hexdigest()))
    inputs=torch.cat(inputs,dim=0)
    output=dict(checkpoint=workload['checkpoint'],checkpoint_sha256=workload['checkpoint_sha256'],inputs=provenance,
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        rows=[],baselines={},note='No blending/accumulation. Same GPU/process/model/input tensors; four selected tiles, full output including borders. nx/ny comparisons include nonlinear direction reconstruction, not just rounding.')
    output_dir=ROOT/'tile_repeatability_float32_u31500_v1' if float32_sweep else OUTPUT
    output_dir.mkdir(parents=True,exist_ok=True)

    def infer(mode,batch,order):
        adapter.config['training']['mixed_precision']=mode
        result=np.empty((4,7,256,256,256),np.float32)
        observed.clear()
        actual=[]
        with torch.inference_mode():
            for offset in range(0,len(order),batch):
                ids=order[offset:offset+batch]
                actual.append(len(ids))
                value=adapter.run_tile_inference(model,inputs[ids],device=device).cpu().numpy()
                assert value.shape==(len(ids),7,256,256,256)
                for j,i in enumerate(ids):
                    result[i]=value[j]
        return result,dict(batch_sizes=actual,model_output_dtypes=sorted(set(observed)))

    baselines={}
    for mode in (('off',) if float32_sweep else ('bf16','fp16','off')):
        batch=1 if mode=='off' else 4
        print(f'Inference-only {mode}: baseline batch{batch}',flush=True)
        base,metadata=infer(mode,batch,list(range(4)))
        encoded=finalize(adapter,base)
        output['baselines'][mode]=metadata
        baselines[mode]=(base,encoded)
        matrix=[('repeat1',batch,list(range(4))),('repeat2',batch,list(range(4)))]
        if mode!='off':
            matrix += [(f'batch{b}',b,list(range(4))) for b in (1,2,3)]
            matrix += [('reverse_batch4',4,[3,2,1,0]),('repeat_after_shapes',4,list(range(4)))]
        elif float32_sweep:
            matrix += [(f'batch{b}',b,list(range(4))) for b in (2,3,4)]
            matrix += [('reverse_batch1',1,[3,2,1,0]),('repeat_after_shapes',1,list(range(4)))]
        for name,b,order in matrix:
            started=time.monotonic()
            print(f'Inference-only {mode}: {name}',flush=True)
            raw,metadata=infer(mode,b,order)
            rounded=finalize(adapter,raw)
            metrics=compare(base,raw,encoded,rounded)
            row=dict(mode=mode,name=name,kind='same_precision',**metadata,**metrics,wall_seconds=time.monotonic()-started)
            output['rows'].append(row)
            atomic_write(output_dir/'results.partial.json',output,durable=False)
            print(f"  raw presence changed={metrics['raw']['6']['changed_percent']:.6f}% rounded={metrics['rounded']['presence']['changed_percent']:.6f}%",flush=True)
            del raw,rounded
    for mode in (() if float32_sweep else ('fp16','off')):
        a,qa=baselines['bf16'];b,qb=baselines[mode]
        output['rows'].append(dict(mode=mode,name='versus_bf16_baseline',kind='cross_precision',**compare(a,b,qa,qb)))
    for i,row in enumerate(provenance):
        assert hashlib.sha256(inputs[i:i+1].cpu().numpy().tobytes()).hexdigest()==row['normalized_sha256']
    hook.remove()
    output['completed']=True
    atomic_write(output_dir/'results.json',output,durable=False)
    print(f'COMPLETE: {output_dir}/results.json',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--unmanaged',action='store_true')
    parser.add_argument('--devices')
    parser.add_argument('--float32-sweep',action='store_true')
    args=parser.parse_args()
    if args.unmanaged:
        assert args.devices=='0'
        worker(args.float32_sweep)
    else:
        from vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation import launch_managed
        command=[sys.executable,'-u',str(Path(__file__).resolve())]
        if args.float32_sweep:
            command.append('--float32-sweep')
        raise SystemExit(launch_managed(command,devices='0',
            settings=dict(exclusive=True,priority=100,cpu_cores=32,label='reviewed-tile-repeatability')))


if __name__=='__main__':
    main()
