"""Paired native CT sampled on the original fine-teacher UV/Z lattice."""
import json
from functools import lru_cache
from scipy.interpolate import RBFInterpolator
import numpy as np
import torch
from scipy.ndimage import gaussian_filter,map_coordinates
from vesuvius.ink_detection.data.multiteacher import stable_seed
from vesuvius.ink_detection.volume_io import open_volume

def appearance(ct,rng,recipe,threshold):
    """One volume-wide contrast draw, never applied to teacher CT or targets."""
    if rng.random()<recipe['identity_probability']:
        return ct.copy(),np.zeros(3,dtype=np.float32),1.
    sigma=np.minimum(rng.uniform(*recipe['blur_sigma'])*rng.uniform(.85,1.15,3),recipe['blur_sigma'][1])
    blurred=gaussian_filter(ct,sigma,mode='reflect')
    tissue=blurred[blurred>threshold]
    pivot=float(np.median(tissue)) if tissue.size else float(np.median(blurred))
    gain=float(rng.uniform(*recipe['contrast']))
    result=np.maximum(pivot+gain*(blurred-pivot),0).astype(np.float32)
    result[ct==0]=0
    return result,sigma.astype(np.float32),gain

@lru_cache(maxsize=32)
def field_predictor(serialized):
    m=json.loads(serialized);p=m['parameters']
    if m['kind']=='translation':return lambda q:np.tile(p['constant'],(len(q),1))
    if m['kind']=='affine':return lambda q:np.c_[np.ones(len(q)),q]@np.array(p['coefficients'])
    return RBFInterpolator(np.array(p['points']),np.array(p['shifts']),kernel=p['kernel'],smoothing=p['smoothing'],degree=p['degree'])

def sample_native(array,recipe,record,yx,reverse,valid_array=None):
    depth=int(record['shape'][0]);indices=np.arange(depth)[::-1] if reverse else np.arange(depth)
    start=depth//2-32;centers=indices[start:start+64].reshape(16,4).mean(1)
    shift=recipe['sampling_shift_zyx']
    z=(centers-(depth-1)/2)*recipe['fine_spacing_um']/recipe['native_depth_step_um']+(array.shape[0]-1)/2+shift[0]
    fy=int(yx[0])+4*np.arange(64)+1.5;fx=int(yx[1])+4*np.arange(64)+1.5
    y=fy*recipe['uv_scale_yx'][0]+shift[1];x=fx*recipe['uv_scale_yx'][1]+shift[2]
    global_coords=np.asarray(np.meshgrid(z,y,x,indexing='ij'))
    if model:=recipe.get('residual_field'):
        uv=np.stack(np.meshgrid(fy,fx,indexing='ij'),-1).reshape(-1,2)
        q=(uv-model['center_fine_yx'])/model['scale_fine_yx']
        field=field_predictor(json.dumps(model,sort_keys=True))(q).reshape(64,64,3).transpose(2,0,1)
        factors=np.array([(-1 if reverse else 1)*4*recipe['fine_spacing_um']/recipe['native_depth_step_um'],4*recipe['uv_scale_yx'][0],4*recipe['uv_scale_yx'][1]])
        global_coords+=field[:,None,:,:]*factors[:,None,None,None]
    if not np.isfinite(global_coords).all():return np.zeros((16,64,64),np.float32),np.zeros((16,64,64),np.float32)
    starts=[max(0,min(n-1,int(np.floor(a.min())))) for a,n in zip(global_coords,array.shape)]
    stops=[max(s+1,min(n,int(np.ceil(a.max()))+1)) for a,n,s in zip(global_coords,array.shape,starts)]
    if np.prod(np.asarray(stops)-starts)>16*128*128:return np.zeros((16,64,64),np.float32),np.zeros((16,64,64),np.float32)
    sl=tuple(slice(a,b) for a,b in zip(starts,stops))
    coords=global_coords-np.asarray(starts)[:,None,None,None]
    ct=map_coordinates(np.asarray(array[sl],dtype=np.float32),coords,order=1,mode='constant',cval=0,prefilter=False)
    inside=np.ones(ct.shape,dtype=bool)
    for a,n in zip(global_coords,array.shape):inside&=(a>=0)&(a<=n-1)
    if valid_array is not None:
        support=map_coordinates(np.asarray(valid_array[sl],dtype=np.float32),coords,order=1,mode='constant',cval=0,prefilter=False)
        inside&=support>=.999
    ct[~inside]=0
    return ct,inside.astype(np.float32)

def ct_correlation(fine,native,valid):
    # Fixed metric: no ink labels, no local shift optimization, before augmentation.
    core=(slice(4,12),slice(8,56),slice(8,56))
    mask=valid[core]>.999
    if mask.mean()<.8 or mask.sum()<1000:return -1.
    a=gaussian_filter(fine.astype(np.float32),.65,mode='reflect')[core][mask].astype(float)
    b=gaussian_filter(native.astype(np.float32),.65,mode='reflect')[core][mask].astype(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.std(),b.std())<1.:return -1.
    a-=a.mean();b-=b.mean()
    score=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return score if np.isfinite(score) else -1.

class PairedInputs:
    def __init__(self,recipe,validation):
        self.recipe=recipe
        self.pairs=json.load(open(recipe['manifest']))
        self.validation=validation
        self.cache={}
        self.quality=json.load(open(recipe['quality_file'])) if recipe.get('quality_file') else None
    def apply(self,sample,record,draw):
        recipe=self.recipe
        rng=np.random.default_rng(stable_seed(recipe['seed'],'paired-input',int(draw),self.validation))
        key=record['scroll']+'/'+record['segment'];pair=self.pairs.get(key)
        score=-1.;eligible=pair is not None
        if pair is not None and self.quality is not None:
            coordinate=','.join(str(int(v)) for v in sample['yx'])
            score=self.quality[key][coordinate]
            eligible=score>recipe['minimum_correlation']
        native=eligible and (self.validation or rng.random()<recipe['native_probability'])
        if self.validation and recipe.get('force_fine_validation'):native=False
        raw=sample['raw'][0].numpy()
        if native:
            path=pair['path']
            if path not in self.cache:
                a=open_volume(path,0);valid=None
                if pair['valid_array']:
                    import zarr
                    valid=zarr.open_group(path,mode='r')[pair['valid_array']]
                self.cache[path]=(a,valid)
            a,v=self.cache[path]
            low,valid=sample_native(a,pair,record,sample['yx'],bool(sample['reverse_depth']),v)
        else:
            low=raw.reshape(16,4,64,4,64,4).mean((1,3,5))
            valid=np.ones(low.shape,dtype=np.float32)
        sigma=np.zeros(3,dtype=np.float32);gain=1.
        if not self.validation:
            aug=recipe['native_augmentation'] if native else recipe
            threshold=0 if native else recipe['background_thresholds'][record['scroll']]
            low,sigma,gain=appearance(low,rng,aug,threshold)
            low[valid==0]=0
        sample['student_input']=torch.from_numpy(np.ascontiguousarray(low/255)).unsqueeze(0)
        sample['student_valid']=torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0)
        if not self.validation:
            rotation=int(rng.integers(4));flip=bool(rng.integers(2))
            for k in ('raw','teacher_image','valid_3d','labels_2d','mask_2d','student_input','student_valid'):
                a=torch.rot90(sample[k],rotation,(-2,-1))
                sample[k]=torch.flip(a,(-1,)).contiguous() if flip else a.contiguous()
        sample['native_domain']=torch.tensor(float(native))
        sample['paired_available']=torch.tensor(pair is not None)
        sample['native_eligible']=torch.tensor(bool(eligible))
        sample['alignment_correlation']=torch.tensor(score)
        sample['blur_sigma']=torch.tensor(sigma)
        sample['contrast_gain']=torch.tensor(gain)
        return sample

def batch_input(batch,factor):
    from vesuvius.ink_detection.data.resolution_distillation import student_input
    return batch['student_input'] if 'student_input' in batch else student_input(batch['raw'],factor)

def mask_pair_targets(targets,batch):
    if 'student_valid' in batch:
        valid=batch['student_valid']
        targets['volume_weight']=targets['volume_weight']*valid
        targets['projection_weight']=targets['projection_weight']*valid.any(2)
    return targets
