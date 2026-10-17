"""Record CPU edge-case output digests, or require exact equality to a prior build.

Run through run_native.py: check_augment_cpu_outputs.py output.json [baseline.json].
Covers singleton axes, fractional/wide radii, coarse/fine lattice caching, high
fractal octaves, signed/oblique blur directions, float32/64 and repeated seeds.
"""
import hashlib,json,sys
import numpy as np
from vc_reader import augment as A
result={}
for shape in [(1,1,1),(1,3,37),(9,11,35)]:
 for dtype in (np.float32,np.float64):
  source=np.random.default_rng(82).uniform(-1,2,(2,*shape)).astype(dtype)
  transforms=[A.Defocus(radius=r) for r in (0,.1,1.5,2,5)]
  transforms += [A.RingingOvershoot(cutoff=c,radius=r) for c in (.1,.7,1) for r in (1,4,12)]
  transforms += [A.BiasField(std=.3,scale=s) for s in (.025,.9,1)]
  transforms += [t(scale=s,octaves=o) for t in (A.PlasmaShadow,A.PlasmaBrightnessContrast) for s,o in ((.025,5),(.7,1),(1,16))]
  transforms += [A.OrientedBlur(axis=axis,sigma_along=1.7,sigma_across=.6) for axis in ((1,0,0),(-1,0,0),(0,1,0),(0,0,-1),(1,.2,0))]
  transforms += [A.GibbsRinging(cutoff=.7),A.FDA(source*.8,blend=.6,fraction=.3)]
  for j,t in enumerate(transforms):
   p=A.Pipeline([t],backend='cpu')
   for seed in (0,123):
    out=p(dict(image=source),seed=seed)['image']
    result[f'{shape}/{dtype}/{j}/{seed}']=hashlib.sha256(out.tobytes()).hexdigest()
if len(sys.argv)>2:
 old=json.load(open(sys.argv[2]));bad=[k for k in result if result[k]!=old[k]]
 assert not bad,bad
 print(len(result),'bitwise-equal outputs')
json.dump(result,open(sys.argv[1],'w'),indent=2)
