#!/usr/bin/env python3
"""Bounded CPU/MPS nnU-Net segmentation adapter for Dataset 058."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np

MODEL_ID = "vc-surface-nnunet-058"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_volume(path: Path) -> np.ndarray:
    import tifffile
    if path.is_dir():
        fs=sorted([*path.glob('*.tif'),*path.glob('*.tiff'),*path.glob('*.png')])
        if not fs: raise ValueError('volume_path contains no TIFF/PNG slices')
        a=np.stack([tifffile.imread(f) for f in fs])
    elif path.suffix == '.npy': a=np.load(path,allow_pickle=False)
    else: a=tifffile.imread(path)
    a=np.asarray(a)
    if a.ndim != 3: raise ValueError(f'expected ZYX volume, got {a.shape}')
    if any(n>256 for n in a.shape) or a.size>16_777_216: raise ValueError('volume exceeds bounded 256^3/16M voxel limit')
    return a

def starts(size:int,tile:int,step:int):
    if size<=tile:return [0]
    out=list(range(0,size-tile+1,step)); last=size-tile
    if out[-1]!=last:out.append(last)
    return out

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=('cpu','mps'),required=True);a=p.parse_args()
    req=json.loads(Path(a.request).read_text()); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    model_dir=Path(req['model_dir']).resolve(); checkpoint=model_dir/'fold_0/checkpoint_best.pth'
    if req.get('model',MODEL_ID)!=MODEL_ID: raise ValueError('unsupported model')
    if sha256(checkpoint)!=req['checkpoint_sha256']: raise ValueError('checkpoint SHA-256 mismatch')
    volume=load_volume(Path(req['volume_path']).resolve())
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    if a.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS is unavailable')
    torch.set_num_threads(max(1,int(req.get('cpu_threads',4))))
    plans=json.loads((model_dir/'plans.json').read_text()); dataset=json.loads((model_dir/'dataset.json').read_text()); cfg=plans['configurations']['3d_fullres']; arch=cfg['architecture']
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=get_network_from_plans(arch['network_class_name'],arch['arch_kwargs'],arch['_kw_requires_import'],len(dataset['channel_names']),len(dataset['labels']),allow_init=False,deep_supervision=False)
    net.load_state_dict(ckpt['network_weights'],strict=True);device=torch.device(a.device);net.eval().to(device)
    props=plans['foreground_intensity_properties_per_channel']['0']; x=np.clip(volume.astype(np.float32),props['percentile_00_5'],props['percentile_99_5']);x=(x-props['mean'])/max(props['std'],1e-6)
    tile=int(req.get('tile_size',64)); overlap=float(req.get('overlap',0.5))
    if tile not in (64,96,128) or any(n%32 for n in (tile,)): raise ValueError('tile_size must be 64, 96, or 128')
    step=max(1,int(tile*(1-overlap))); pads=[max(0,tile-n) for n in x.shape];xp=np.pad(x,tuple((0,v) for v in pads),mode='edge')
    logits=np.zeros((2,*xp.shape),np.float32);weights=np.zeros(xp.shape,np.float32)
    zs,ys,xs=(starts(n,tile,step) for n in xp.shape); total=len(zs)*len(ys)*len(xs); done=0
    with torch.inference_mode():
        for z in zs:
            for y in ys:
                for xx in xs:
                    patch=torch.from_numpy(xp[z:z+tile,y:y+tile,xx:xx+tile].copy())[None,None].to(device)
                    pred=net(patch)[0].float().cpu().numpy();logits[:,z:z+tile,y:y+tile,xx:xx+tile]+=pred;weights[z:z+tile,y:y+tile,xx:xx+tile]+=1;done+=1;print(f'progress {done}/{total}',flush=True)
    logits/=np.maximum(weights,1e-6)[None]; logits=logits[:,:volume.shape[0],:volume.shape[1],:volume.shape[2]]
    exp=np.exp(logits-logits.max(0,keepdims=True));prob=exp[1]/exp.sum(0);threshold=float(req.get('threshold',0.5));mask=(prob>=threshold).astype(np.uint8)
    np.save(out/'surface-probability.npy',prob.astype(np.float32));np.save(out/'surface-mask.npy',mask)
    import tifffile
    tifffile.imwrite(out/'surface-probability.tif',prob.astype(np.float32));tifffile.imwrite(out/'surface-mask.tif',mask)
    from PIL import Image
    Image.fromarray(np.clip(prob.max(0)*255,0,255).astype(np.uint8)).save(out/'probability-preview.png');Image.fromarray(mask.max(0)*255).save(out/'mask-preview.png')
    manifest={'kind':'nnunet_volume_segmentation_v1','model':MODEL_ID,'backend':a.device,'checkpoint_sha256':req['checkpoint_sha256'],'input':str(req['volume_path']),'input_shape_zyx':list(volume.shape),'tile_size':tile,'overlap':overlap,'threshold':threshold,'patches':total,'class_semantics':{'0':'background','1':'fiber/surface'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())
