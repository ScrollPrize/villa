"""Controlled paired identity overfit: valid acceptance and wrong rejection both required."""
import argparse
import json
from pathlib import Path

import torch

from .model import DirectConfig, build_model
from .supervision import loss_terms
from .history_diagnostic import paired_history_report
from .train import move_batch
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


def controlled_pair(cfg):
    b, k = 4, cfg.n_future
    image = torch.rand(1,cfg.input_channels,cfg.fine.depth,cfg.fine.width,cfg.fine.width).expand(b,-1,-1,-1,-1).clone()
    slabs = torch.zeros(b,8,2,8,65,65)
    # CT strand context differs; the observed strand marker and all metadata
    # stay fixed. Current crops, local histories and candidate order are identical.
    slabs[0,0,0,:,:,18:24] = 1.
    slabs[1,0,0,:,:,40:46] = 1.
    slabs[2,1,0,:,:,18:24] = 1.
    slabs[3,1,0,:,:,40:46] = 1.
    slabs[:,:2,1,:,:,31:34] = 1.
    valid = torch.zeros(b,8,dtype=torch.bool);valid[:,:2]=True
    pose = torch.zeros(b,8,14);pose[:,0,2]=-2.;pose[:,0,12]=.7;pose[:,0,13]=1.
    x = dict(fine=image,history_slabs=slabs,history_valid=valid,history_pose=pose,
             history_ages=torch.full((b,8),256.),history_overlap=torch.zeros(b,8),history_load_seconds=torch.zeros(b))
    curves = torch.zeros(b,2,k,3)
    curves[:,:,:,2] = torch.arange(1,k+1)*cfg.future_step
    curves[:,0,:,0] = -2.;curves[:,1,:,0] = 2.
    labels = torch.zeros(b,2,k);labels[::2,0]=1.;labels[1::2,1]=1.
    dense = torch.zeros(b,4*(k-1)+1,2);dense[::2,:,0]=-2.;dense[1::2,:,0]=2.
    return dict(x=x,hist=torch.zeros(b,cfg.n_history,3),hmask=torch.zeros(b,cfg.n_history),
                dense_ab=dense,dense_mask=torch.ones(dense.shape[:2]),offtrack=torch.zeros(b),
                endpoint_known=torch.zeros(b),end_local=torch.zeros(b,3),source=torch.zeros(b),
                candidate_points=curves,candidate_labels=labels,candidate_mask=torch.ones_like(labels))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--steps',type=int,default=250)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output path')
    torch.set_num_threads(2);torch.manual_seed(491)
    cfg=DirectConfig(fine=CropSpec(24,17,8,1.),channels=4,hidden=32,heads=4,layers=1,
                     decoder_layers=2,n_history=32,n_future=4,recurrent_refinement_steps=1)
    model=build_model(cfg).to(args.device)
    cpu=controlled_pair(cfg);batch=move_batch(cpu,args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    losses=[]
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out=model(batch['x'],batch['hist'],batch['hmask'],candidates=batch['candidate_points'])
        terms=loss_terms(out,batch,cfg)
        loss=terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()+terms['candidate_per_state'].mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),100.);opt.step()
        if step%25==0:
            losses.append(dict(step=step,loss=float(loss.detach())))
            print(json.dumps(losses[-1]),flush=True)
    report=paired_history_report(model,batch)
    with torch.no_grad():
        prediction=model(batch['x'],batch['hist'],batch['hmask'])
    target=batch['candidate_points'][torch.arange(4,device=args.device),torch.arange(4,device=args.device)%2]
    geometry_error=float((prediction['points']-target).norm(dim=-1).mean())
    generated_accepted=int((prediction['confidence'][:,-1]>=.5).sum())
    full=report['full']
    passed=full['geometry_choice_accuracy']==1. and full['correct_acceptance']==1. and full['wrong_rejection']==1. and geometry_error<.25 and generated_accepted==4
    result=dict(passed=passed,geometry_error=geometry_error,generated_accepted=generated_accepted,steps=args.steps,seed=491,device=args.device,losses=losses,ablations=report)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    torch.save(cpu,args.out.with_suffix('.fixture.pt'))
    print(json.dumps(result,indent=2))
    if not passed:
        raise RuntimeError('Controlled identity overfit failed; rejecting everything is insufficient')


if __name__=='__main__':main()
