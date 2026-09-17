"""Training-only activation-statistic calibration for frozen EMA snapshots."""
from contextlib import nullcontext
import hashlib

import torch
from torch import nn


def batchnorm_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name.endswith(('running_mean','running_var','num_batches_tracked')):
            value=value.detach().cpu().contiguous()
            digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def recalibrate_batchnorm(model, batches, forward, *, autocast=nullcontext):
    """Change BN buffers only; restore RNG, module modes and momentum on exit.

    Every distributed rank must consume the same number of batches when the
    model contains SyncBatchNorm. `forward(model, batch)` handles preprocessing.
    """
    from vesuvius.ink_detection.training.multiteacher import rng_state,restore_rng
    bns=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)
         and m.track_running_stats]
    if not bns:raise ValueError('EMA calibration requires tracked BatchNorm')
    modes=[(m,m.training) for m in model.modules()]
    saved=[(m,m.momentum,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone()) for m in bns]
    rng=rng_state();count=0
    try:
        model.eval()
        for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
        with autocast():
            for batch in batches:
                forward(model,batch);count+=1
        if not count:raise ValueError('Empty EMA calibration stream')
        for bn in bns:
            if not torch.isfinite(bn.running_mean).all() or not torch.isfinite(bn.running_var).all():
                raise FloatingPointError('Non-finite EMA BatchNorm statistics')
    except BaseException:
        for bn,_,mean,var,tracked in saved:
            bn.running_mean.copy_(mean);bn.running_var.copy_(var);bn.num_batches_tracked.copy_(tracked)
        raise
    finally:
        for bn,momentum,*_ in saved:bn.momentum=momentum
        for module,mode in modes:module.training=mode
        restore_rng(rng)
    return {'batches_per_rank':count,'batchnorm_modules':len(bns),
            'buffer_sha256':batchnorm_digest(model.state_dict())}


def verify_calibrated_ema(payload):
    """Reject stale/uncalibrated resolution-distillation teachers."""
    cfg=payload['config'];recipe=cfg.get('ema_batchnorm_calibration')
    record=payload.get('ema_batchnorm_calibration')
    if (not recipe or not record or record.get('method')!='training_only_cumulative'
            or record.get('optimizer_update')!=payload['optimizer_step']
            or record.get('recipe')!=recipe
            or record.get('manifest_sha256')!=cfg.get('manifest_sha256')
            or record.get('patches_sha256')!=cfg.get('patches_sha256')
            or record.get('buffer_sha256')!=batchnorm_digest(payload['ema_model'])):
        raise ValueError('Require a current training-calibrated EMA with verified BatchNorm buffers')
