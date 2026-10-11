"""GPU-managed equivalence, timing and profiling of the shared UV pair loss."""
import json
import time
from pathlib import Path
import numpy as np
from . import fiber_uv_layout as uv


def benchmark(config):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Pair benchmark requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'benchmark.json').exists():raise FileExistsError('Completed benchmark exists')
    with np.load(config['tensor_cache']) as src:d={k:src[k] for k in src.files}
    table=torch.as_tensor(uv.make_distance_table(float(d['distance'].max()),config.get('distance_sigma',.5)),device='cuda')
    result=dict(config=config,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),cases=[])
    for case in config.get('benchmark_states',[{'label':'initial'}]):
        case_config=dict(config,**case.get('options',{}))
        eager=uv.pair_evaluator(dict(case_config,pair_backend='eager'))
        compiled=uv.pair_evaluator(dict(case_config,pair_backend='compiled'))
        if case.get('path'):
            with np.load(case['path']) as src:state=src['state'];widths=src['widths']
        else:state,widths=uv.fresh_initial_state(d,config)
        k=len(widths);w=torch.as_tensor(widths,device='cuda');bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
        count=int(case.get('batch_size',config.get('batch_size',4096)))
        edges=np.random.default_rng(0).choice(len(d['i']),count,replace=False)
        def tensor(value,dtype=torch.float32):return torch.as_tensor(value,device='cuda',dtype=dtype)
        def endpoint(key,ds_key):
            ids=d[key][edges];s=tensor(state[ids]);p=s[:,:k+1].softmax(-1)
            _,v,q=uv.coordinates(s,tensor(d['phase'][ids]),tensor(d['z'][ids]),
                tensor(d['family'][ids],torch.long),tensor(d[ds_key][edges]),w,bounds)
            return p[:,:k].detach().requires_grad_(),q.detach().requires_grad_(),v.detach().requires_grad_()
        pa,qa,va=endpoint('i','ds_a');pb,qb,vb=endpoint('j','ds_b')
        inputs=[pa,pb,qa,qb,va,vb]
        fixed=[tensor(d['phase_delta'][edges]),tensor(d['family'][d['j'][edges]]-d['family'][d['i'][edges]],torch.long),
            tensor(d['sign'][edges]),tensor(d['distance'][edges]),table,case_config]
        if case.get('boundaries'):
            with torch.no_grad():
                qa.copy_(torch.arange(k,device='cuda')[None,:]);qb.copy_(qa)
                va.zero_();vb.fill_(1280.)
                fixed[0].zero_();fixed[1].zero_();fixed[2].copy_(tensor(np.resize([-1,0,1],count)))
                fixed[3].fill_(.5)
                # Exact zero/half-turn/bin boundaries and >104 lookup extrapolation.
                qb[::2].add_(.5)
        def call(fn):
            for x in inputs:x.grad=None
            value=fn(*inputs,*fixed);value.backward();return value
        records={};reference=None
        for name,fn in [('eager',eager),('compiled',compiled)]:
            torch.cuda.synchronize();started=time.monotonic();value=call(fn);torch.cuda.synchronize()
            first=time.monotonic()-started
            values=[float(value.detach())]+[x.grad.detach().cpu().numpy().copy() for x in inputs]
            if reference is None:reference=values
            else:
                np.testing.assert_allclose(values[0],reference[0],rtol=2e-5,atol=2e-5)
                for a,b in zip(values[1:],reference[1:]):np.testing.assert_allclose(a,b,rtol=2e-3,atol=2e-6)
            errors=[]
            for a,b in zip(values[1:],reference[1:]):
                errors.append(dict(max_absolute=float(np.max(np.abs(a-b))),
                    relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))))
            del value
            for _ in range(3):call(fn)
            for x in inputs:x.grad=None
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
            elapsed=[]
            for _ in range(int(config.get('benchmark_repeats',20))):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();call(fn);end.record();end.synchronize();elapsed.append(begin.elapsed_time(end))
            peak=torch.cuda.max_memory_allocated();torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):call(fn)
                torch.cuda.synchronize()
            kernels=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            top=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)[:15]
            records[name]=dict(first_forward_backward_seconds=first,loss=values[0],gradient_errors=errors,
                mean_ms=float(np.mean(elapsed)),p50_ms=float(np.quantile(elapsed,.5)),
                p95_ms=float(np.quantile(elapsed,.95)),min_ms=min(elapsed),max_ms=max(elapsed),
                repeats=len(elapsed),peak_allocated_bytes=peak,incremental_peak_bytes=peak-base,
                cuda_events_per_call=len(kernels)/3,
                profile=[dict(name=e.key,self_cuda_us=e.self_device_time_total,count=e.count) for e in top])
            print('PAIR_BENCH '+json.dumps(dict(case=case['label'],backend=name,**records[name])),flush=True)
        result['cases'].append(dict(label=case['label'],batch_size=count,winding_count=k,
            input_strides=[list(x.stride()) for x in inputs],results=records,
            speedup=records['eager']['mean_ms']/records['compiled']['mean_ms']))
        uv.atomic_write(root/'benchmark.json',result,durable=False)
    uv.atomic_write(root/'summary.json',result,durable=False)
    return result
