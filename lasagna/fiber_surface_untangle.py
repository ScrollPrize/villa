"""Explicit signed bilinear-Jacobian fitting and feasible joint step acceptance."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from opt_loss_surface_guard import corner_vectors
from opt_loss_fixed_uv import FixedUVTargets


def signed_areas(x, normal, step):
    # One fixed normal per cell: projected bilinear Jacobian is affine in UV.
    return torch.stack([(c*normal).sum(-1)/step**2 for c in corner_vectors(x)])


def area_stats(x, normal, step, margin=.001):
    with torch.no_grad():
        j = signed_areas(x, normal, step)
        return dict(nonpositive_corners=int((j<=0).sum()), below_margin_corners=int((j<margin).sum()),
                    bad_cells=int((j<margin).any(0).sum()), minimum=float(j.min()),
                    quantiles=[float(torch.quantile(j.flatten(),q)) for q in (.001,.01,.5)],
                    deficit=float(torch.relu(margin-j).sum()))


def normal_field(prior, attachments, initial, normal_root, sigma=2.):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_surface import CurveBank
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_convention import inward_solution
    config = json.loads((Path(prior['settings']['run_root'])/'config.json').read_text())
    with np.load(config['stage1']) as f:
        sign = inward_solution({k:f[k] for k in f.files})
    bank = CurveBank(config['afv'], attachments['source'], normal_root, sign)
    _, normal, valid = bank.sample(np.searchsorted(bank.ids,attachments['source']),attachments['arc'])
    if not valid.any():
        raise ValueError('No valid aligned fiber normals')
    h,w = initial.shape[:2]
    y,x = np.mgrid[:h-1,:w-1]
    positions = np.column_stack([x.ravel()+.5,y.ravel()+.5])
    distance, ids = cKDTree(attachments['final_grid_uv'][valid]).query(positions,workers=8)
    nearest = normal[valid][ids].reshape(h-1,w-1,3)
    field = ndimage.gaussian_filter(nearest, (sigma,sigma,0))
    length = np.linalg.norm(field,axis=-1)
    low = length<.25
    field[low] = nearest[low]  # Preserve a measured orientation, never a zero vector.
    field /= np.linalg.norm(field,axis=-1,keepdims=True)
    # Only global chart parity is inferred from the mesh, not local orientation.
    corners = corner_vectors(torch.tensor(initial,dtype=torch.float64))
    coarse = sum(corners).numpy()
    agreement = (coarse*field).sum(-1)
    near = distance.reshape(h-1,w-1)<2
    parity = 1 if np.median(agreement[near])>=0 else -1
    field *= parity
    info = dict(valid_normal_samples=int(valid.sum()), invalid_normal_samples=int((~valid).sum()),
                chart_parity=parity, cancellation_cells=int(low.sum()),
                nearest_support_distance_grid_p50_p95_max=np.quantile(distance,[.5,.95,1]).tolist(),
                farther_than_10_cells=int((distance>10).sum()),
                normal_root=str(normal_root), normal_sign_solution=config['stage1'])
    return field.astype(np.float32),info


def untangle(initial, targets, normal, step, *, halos=(8,16,32), steps=1000, lr=.02,
             max_patches=None, margin=.001, guide_weight=.001, proximity_weight=.0001,
             dual_weight=0.):
    device = targets.xyz.device
    x = torch.as_tensor(initial,device=device).clone()
    n = torch.as_tensor(normal,device=device)
    h,w = initial.shape[:2]
    # Frozen UV; use shared precomputed bilinear sampling, not a new data loss.
    frozen = FixedUVTargets(targets.current_uv().detach(),targets.xyz,targets.arc_weights,(h,w),device=device)
    if hasattr(targets, 'data_weights'):
        frozen.arc_weights = targets.data_weights.detach().clone()
    history=[]
    for halo in halos:
        bad = (signed_areas(x,n,step)<margin).any(0).cpu().numpy()
        if not bad.any(): break
        labels,count = ndimage.label(ndimage.binary_dilation(bad,iterations=halo),np.ones((3,3)))
        order=np.argsort(-np.bincount(labels.ravel())[1:],kind='stable')+1
        if max_patches is not None: order=order[:max_patches]
        cells=np.isin(labels,order)
        vertex=np.zeros((h,w),bool)
        for dy,dx in [(0,0),(0,1),(1,0),(1,1)]: vertex[dy:dy+h-1,dx:dx+w-1]|=cells
        free=ndimage.binary_erosion(vertex,np.ones((3,3)))
        indices=torch.as_tensor(np.flatnonzero(free),device=device)
        if not len(indices): continue
        baseline=x.clone()
        position=torch.nn.Parameter((baseline.reshape(-1,3)[indices]/step).clone())
        opt=torch.optim.Adam([position],lr=lr)
        multiplier=torch.zeros_like(signed_areas(x,n,step))
        best=x.clone(); best_j=signed_areas(x,n,step)
        best_key=(float(torch.relu(margin-best_j).square().sum()),int((best_j<margin).sum()))
        start_error=float(((frozen.predict(x[None])-targets.xyz).norm(dim=-1)*targets.arc_weights).sum())
        for iteration in range(steps):
            trial=baseline.reshape(-1,3).index_copy(0,indices,position*step).reshape(h,w,3)
            j=signed_areas(trial,n,step)
            # Do NOT normalize by entire mesh area: local inversions must dominate.
            loss=torch.relu(margin*10-j).square().sum()/max(1,int(bad.sum()))
            if dual_weight:
                loss=loss+(multiplier*torch.relu(margin*10-j)).sum()/max(1,int(bad.sum()))
            fit=((frozen.predict(trial[None])-targets.xyz).square().sum(-1)*targets.arc_weights).sum()/step**2
            proximity=(position-baseline.reshape(-1,3)[indices]/step).square().mean()
            loss=loss+guide_weight*fit+proximity_weight*proximity
            opt.zero_grad();loss.backward();opt.step()
            if dual_weight and iteration%20==0:
                with torch.no_grad(): multiplier.add_(dual_weight*torch.relu(margin*10-j)).clamp_(max=1000)
            if iteration%100==0 or iteration==steps-1:
                with torch.no_grad():
                    trial=baseline.reshape(-1,3).index_copy(0,indices,position*step).reshape(h,w,3)
                    j=signed_areas(trial,n,step)
                    error=float(((frozen.predict(trial[None])-targets.xyz).norm(dim=-1)*targets.arc_weights).sum())
                    key=(float(torch.relu(margin-j).square().sum()),int((j<margin).sum()))
                    # Large movement is permitted to untangle, but not unbounded
                    # sacrifice of the observed fibers. Never hide rejected trials.
                    admissible=bool(torch.isfinite(trial).all()) and error<=start_error+5
                    if admissible and key<best_key: best=trial.clone();best_key=key
                    row=dict(halo=halo,iteration=iteration+1,loss=float(loss),mean_error=error,
                             admissible=admissible,**area_stats(trial,n,step,margin))
                    history.append(row);print('untangle '+json.dumps(row),flush=True)
        x=best
    return x.detach().cpu().numpy(),history


class FeasibleStep:
    """Rollback joint parameters; accept only positive protected Jacobians.

    Adam moments advance for damped steps, reset on a fully rejected step. This is
    safeguarded Adam, not an exact line-search optimizer or checkpoint resume.
    """
    def __init__(self,initial,normal,step,margin=.001,max_backtracks=16):
        self.normal,self.step,self.margin=normal,step,margin
        self.protected=signed_areas(initial,normal,step)>=margin
        self.max_backtracks=max_backtracks
        self.accepted=self.damped=self.rejected=0
        self.smallest_fraction=1.

    def before(self,optimizer,model):
        return [(p,p.detach().clone()) for group in optimizer.param_groups for p in group['params']]

    @torch.no_grad()
    def after(self,optimizer,model,saved):
        proposed=[p.detach().clone() for p,_ in saved]
        for attempt in range(self.max_backtracks+1):
            fraction=2.**-attempt
            for (p,old),new in zip(saved,proposed): p.copy_(old+(new-old)*fraction)
            mesh=model._grid_xyz()[0]
            j=signed_areas(mesh,self.normal,self.step)
            if bool(torch.isfinite(mesh).all()) and bool((j[self.protected]>=self.margin).all()):
                self.protected |= j>=self.margin
                self.accepted+=1;self.damped+=int(attempt>0)
                self.smallest_fraction=min(self.smallest_fraction,fraction)
                return
        for p,old in saved: p.copy_(old)
        optimizer.state.clear()
        self.rejected+=1

    def statistics(self):
        return dict(accepted=self.accepted,damped=self.damped,rejected=self.rejected,
                    smallest_fraction=self.smallest_fraction,protected_corners=int(self.protected.sum()))
