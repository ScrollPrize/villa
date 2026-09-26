"""Bounded real-CT joint update, EMA and checkpoint-resume correctness check.

Uses the follower's local level-0 CT and the preview's training fiber. No downloads,
training-run changes, or claims about learned detection accuracy.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import shutil
import time
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.data import load_fibers, SampleConfig, label_state
from vesuvius.neural_tracing.fiber_follow.geometry import interp_at, arclength
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.direct.model import DirectFollower, DirectConfig
from vesuvius.neural_tracing.fiber_follow.direct.data import ObservationBuilder
from vesuvius.neural_tracing.fiber_follow.direct.judge_model import CTJudge, JudgeConfig
from vesuvius.neural_tracing.fiber_follow.direct.judge_slices import SliceConfig
from vesuvius.neural_tracing.fiber_follow.direct.judge_supervision import JointObservationBuilder
from vesuvius.neural_tracing.fiber_follow.direct.train import optimizer_update, save_checkpoint, load_checkpoint
from vesuvius.neural_tracing.fiber_follow.runloop import training_rng_state, resume_training
from vesuvius.neural_tracing.fiber_follow.direct.judge_options import load_judge
from vesuvius.neural_tracing.fiber_follow.direct.judge_policy import JudgePolicyConfig
from vesuvius.neural_tracing.fiber_follow.direct.judge_model import JUDGE_ARCHITECTURE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--ct-array',default='/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/0')
    ap.add_argument('--ct-grid-scale',type=float,default=4.)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'direct/ct_slice_judge_examples/manifest.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy('/mnt/raid_nvme/spiral_dataset_working/fibers/'+manifest['fiber'],tmp)
        fiber, = load_fibers(tmp)
    cfg = DirectConfig()
    slices = SliceConfig(source=args.ct_array, grid_scale=args.ct_grid_scale)
    model = DirectFollower(cfg).to(args.device)
    ema = copy.deepcopy(model).requires_grad_(False)
    judge = CTJudge(JudgeConfig(spacing=slices.spacing, view_batch=3)).to(args.device)
    judge_ema = copy.deepcopy(judge).requires_grad_(False)
    opt = torch.optim.AdamW([dict(params=model.parameters()),dict(params=judge.parameters())],lr=1e-4)
    spec = FiberVolumeSpec('/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs',
                          ct_zarr='/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr',
                          ct_level=0,ct_grid_scale=4.,inputs='ct+presence')
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,future_step=cfg.future_step)
    source = np.load(root/'direct/ct_slice_judge_examples/slices.npz')
    observed = interp_at(fiber.points,fiber.s,np.arange(200.,233.))
    frame = source['frames'][-1,0]
    hist = np.repeat(observed[:1],cfg.n_history,axis=0)
    hist[:32] = observed[:-1][::-1]
    mask = np.zeros(cfg.n_history,np.float32); mask[:32]=1
    item = label_state(fiber,observed[-1],frame,hist,mask,sample,t=232.,reverse=False)
    item.update(source=0,source_step=-1,stratum=-1,judge_context=dict(path=observed,annotation=fiber.points,
                q0=200.,physical_end=fiber.endpoint_stop[1],seed_frame=source['frames'][0,0]))
    builder = JointObservationBuilder(ObservationBuilder(cfg),slices,synthetic_fraction=0)
    begin = time.monotonic()
    batch = builder([item],FiberVolume(spec,cache_bytes=256<<20))
    assert len(batch['judge']) == 1 and batch['judge'][0]['eligible'].any(), 'No supported native CT sequence'
    metrics = optimizer_update(model,ema,opt,[batch],1,1e-4,device=args.device,judge=judge,judge_ema=judge_ema)
    extra = dict(step=1,optimizer=opt.state_dict(),rng=training_rng_state(),judge=judge.state_dict(),
                 judge_ema=judge_ema.state_dict(),judge_architecture=JUDGE_ARCHITECTURE,judge_cfg=judge.cfg.to_dict(),
                 judge_slices=slices.to_dict(),judge_policy=JudgePolicyConfig().to_dict())
    save_checkpoint(args.out/'smoke.pt',model,ema,spec,sample,extra)
    _,_,_,_,ck = load_checkpoint(args.out/'smoke.pt',args.device)
    loaded = load_judge(ck,args.device)['judge']
    for a,b in zip(loaded.parameters(),judge_ema.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    model2,judge2 = DirectFollower(cfg).to(args.device),CTJudge(judge.cfg).to(args.device)
    ema2,je2 = copy.deepcopy(model2),copy.deepcopy(judge2)
    opt2 = torch.optim.AdamW([dict(params=model2.parameters()),dict(params=judge2.parameters())],lr=1e-4)
    judge2.load_state_dict(ck['judge']);je2.load_state_dict(ck['judge_ema'])
    assert resume_training(ck,model2,ema2,opt2)[0] == 1
    second = optimizer_update(model2,ema2,opt2,[batch],2,1e-4,device=args.device,judge=judge2,judge_ema=je2)
    from vesuvius.neural_tracing.fiber_follow.direct.diagnostics import plot_judge_sequence
    seq = batch['judge'][0]
    with torch.no_grad():
        logits=loaded(**{k:seq[k].to(args.device) for k in ('images','metadata','valid','queries')})
    plot_judge_sequence(seq,logits,args.out/'sequence.png')
    result=dict(first=metrics, resumed=second,seconds=time.monotonic()-begin,source=builder.reader.identity,
                scope='Two joint real-CT updates; exact EMA reload; optimizer/RNG resume; no accuracy claim')
    from vesuvius.neural_tracing.fiber_follow.direct.judge_slices import SliceStream
    from vesuvius.neural_tracing.fiber_follow.direct.judge_model import FeatureCache,sequence_tensors
    # CPU wall time, decoded LRU cold/warm (the OS page cache is not flushed).
    timings=[]
    reader=slices.open()
    for mode in ('decoded_cold','decoded_warm','features_warm'):
        start=time.perf_counter()
        if mode!='features_warm':
            stream=SliceStream(reader,slices,source['frames'][0,0])
            cache=FeatureCache()
        records=stream.update(observed)
        sampled=time.perf_counter()
        inputs=sequence_tensors(records,args.device)
        before=cache.encoded_views
        with torch.no_grad():
            tokens=cache.tokens(loaded,records,args.device)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            encoded=time.perf_counter()
            loaded.decode(tokens,inputs['metadata'],inputs['valid'],inputs['queries'])
            if args.device.startswith('cuda'):torch.cuda.synchronize()
        timings.append(dict(mode=mode,sample_seconds=sampled-start,encode_seconds=encoded-sampled,
                            decoder_seconds=time.perf_counter()-encoded,total_seconds=time.perf_counter()-start,
                            new_views=cache.encoded_views-before,raw_bytes=sum(r['images'].nbytes for r in records),
                            token_bytes=tokens.numel()*tokens.element_size()))
    result['inference_timings']=timings
    result['read_stats']=reader.read_stats
    (args.out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
