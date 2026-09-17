"""Prepare, train and export Hecate with the recorded resolution-distillation objective.

python -m vesuvius.ink_detection.training.resolution_distillation prepare CHECKPOINT OUTPUT
python -m vesuvius.ink_detection.training.resolution_distillation smoke CONFIG
Production requires the separate `train CONFIG --production` command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vesuvius.ink_detection.data.paired_native import batch_input, mask_pair_targets
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.data.multiteacher import RankDrawSampler, file_sha256
from vesuvius.ink_detection.data.resolution_distillation import (
    ResolutionDataset, student_input, teacher_targets, distillation_loss,
    weighted_mean_per_sample, foreground_mask)
from vesuvius.ink_detection.models.canonical_projection import canonical_source_dir
from vesuvius.ink_detection.models.hecate_checkpoint import load_hecate, model_config, sampling_from_config, export_ema
from vesuvius.ink_detection.models.checkpoint import load_model_state
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.training.ema_batchnorm import recalibrate_batchnorm,verify_calibrated_ema
from vesuvius.ink_detection.training.multiteacher import (
    update_ema, rng_state, restore_rng, seed_worker, move_batch)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def schedule_scale(step, warmup, total):
    if step < warmup:
        return step/max(warmup, 1)
    return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1,total-warmup))))


def validate_resume_recipe(previous, current, checkpoint, step):
    """Allow a recorded microbatch regrouping, preserving every other recipe field."""
    ignored={'out_dir','wandb_run_name'}
    comparable=lambda c,keys:{k:v for k,v in c.items() if k not in keys}
    if comparable(previous,ignored)==comparable(current,ignored):return False
    if current.get('loss_transitions'):
        history=current['loss_transitions'];old=previous.get('loss_transitions',[])
        tr=history[-1]
        expected=deepcopy(previous);expected['loss_weight_3d']=tr.get('to_3d')
        if (len(history)!=len(old)+1 or history[:-1]!=old
                or tr.get('from_3d')!=previous['loss_weight_3d']
                or comparable(expected,ignored|{'loss_transitions'})!=comparable(current,ignored|{'loss_transitions'})
                or tr.get('optimizer_update')!=step
                or tr.get('checkpoint_sha256')!=file_sha256(checkpoint)
                or not isinstance(tr.get('to_3d'),(int,float)) or not math.isfinite(tr['to_3d']) or tr['to_3d']<0):
            raise ValueError('Invalid checkpoint-bound 3D loss transition')
        return True
    if current.get('data_transitions'):
        history=current['data_transitions'];old=previous.get('data_transitions',[])
        expected=deepcopy(previous)
        expected['data_config']['paired_native']['native_probability']=1.0
        ignored_data=ignored|{'data_transitions'}
        if (len(history)!=len(old)+1 or history[:-1]!=old
                or comparable(expected,ignored_data)!=comparable(current,ignored_data)):
            raise ValueError('Data transition must only replace paired fine inputs by native inputs')
        transition=history[-1]
        if (transition.get('optimizer_update')!=step
                or transition.get('checkpoint_sha256')!=file_sha256(checkpoint)
                or transition.get('kind')!='native_only_for_paired_segments'):
            raise ValueError('Data transition is not bound to the exact checkpoint')
        return True
    history=current.get('batch_transitions',[])
    old_history=previous.get('batch_transitions',[])
    if len(history)!=len(old_history)+1 or history[:-1]!=old_history:
        raise ValueError('Resume recipe mismatch: require an explicit batch transition')
    transition=history[-1]
    for key in ('batch_size','grad_acc_steps'):
        if (transition.get('from_'+key)!=previous[key] or transition.get('to_'+key)!=current[key]
                or not isinstance(current[key],int) or current[key]<1):
            raise ValueError('Invalid batch transition')
    if (previous['batch_size']*previous['grad_acc_steps']!=current['batch_size']*current['grad_acc_steps']
            or comparable(previous,ignored|{'batch_size','grad_acc_steps','batch_transitions'})
               !=comparable(current,ignored|{'batch_size','grad_acc_steps','batch_transitions'})):
        raise ValueError('Batch transition must preserve effective batch and all other recipe fields')
    if transition.get('optimizer_update')!=step or transition.get('checkpoint_sha256')!=file_sha256(checkpoint):
        raise ValueError('Batch transition is not bound to this exact checkpoint/update')
    return True


def add_code_to_artifact(artifact):
    base=Path(__file__).parents[1]
    for relative in ('training/resolution_distillation.py','data/resolution_distillation.py','data/paired_native.py',
                     'models/canonical_projection.py','models/hecate_checkpoint.py',
                     'models/deterministic_pool.py','data/multiteacher.py',
                     'training/ema_batchnorm.py'):
        artifact.add_file(str(base/relative),name='code/'+relative)
    for relative in ('model_resnet3d_3d_decoder.py', 'models/resnetall.py'):
        artifact.add_file(str(canonical_source_dir()/relative), name='code/canonical/'+relative)


def prepare(checkpoint, output, factor=4, data_config=None, student_checkpoint=None):
    """Create an editable recipe; preparation never starts training.

    A Hub checkpoint contains no dataset paths. Supply the audited data config
    explicitly, or inherit it from a trusted full historical checkpoint.
    """
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    parent = payload['config']
    if sampling_from_config(parent) != 2.4 or 'ema_model' not in payload:
        raise ValueError('Distillation requires a 2.4um Hecate EMA teacher')
    if parent.get('task') == 'resolution_distillation':
        verify_calibrated_ema(payload)
    data = (json.loads(Path(data_config).read_text()) if data_config else
            deepcopy(parent.get('data_config', parent)))
    if 'manifest' not in data or 'segment_depth_reversals' not in data:
        raise ValueError('Supply --data-config with a manifest and explicit segment_depth_reversals')
    thresholds = data.get('background_thresholds', data.get('distillation', {}).get('coarse', {}).get('background_thresholds'))
    if not thresholds:
        raise ValueError('Supply calibrated raw-CT background_thresholds for every scroll')
    if factor not in (1, 4):
        raise ValueError('Factor must be one or four')
    cfg = model_config(2.4 if factor == 1 else 9.6, training=True)
    cfg.update({
        'format_version': 3, 'task': 'resolution_distillation',
        'teacher_checkpoint': str(checkpoint), 'teacher_sha256': file_sha256(checkpoint),
        'teacher_weights': 'ema_model', 'teacher_normalization': parent['image_normalization'],
        'data_config': data, 'background_thresholds': thresholds,
        'manifest_sha256': file_sha256(data['manifest']),
        'patches_sha256': file_sha256(Path(data['manifest']).parent/'patches.npz'),
        'downsample_factor': factor, 'teacher_patch_size': [64, 256, 256],
        'optimizer_updates': 100000, 'warmup_steps': 2000, 'learning_rate': 1e-3,
        'weight_decay': 3e-5, 'grad_clip': 1., 'batch_size': 1, 'grad_acc_steps': 4,
        'expected_world_size': 4, 'mixed_precision': 'bf16', 'seed': 27,
        'ema_decay': .9995, 'sync_batchnorm': True,
        'ema_batchnorm_calibration': {'method': 'training_only_cumulative',
            'global_samples': 512, 'draw_start': 0, 'batch_size_per_gpu': 4},
        'log_live_validation': True, 'loss_weight_2d': .5, 'loss_weight_3d': .5,
        'save_every': 1000, 'val_every': 1000, 'log_every': 20, 'dataloader_workers': 4,
        'out_dir': str(output/'run'), 'wandb_entity': None, 'wandb_project': 'hecate',
        'wandb_run_name': f'hecate_{2.4 if factor == 1 else 9.6}um_distillation',
        'wandb_mode': 'online',
    })
    if student_checkpoint:
        initial, initial_config = load_hecate(student_checkpoint)
        if sampling_from_config(initial_config) != sampling_from_config(cfg):
            raise ValueError('Student checkpoint sampling differs from the requested student')
        del initial
        cfg.update(student_initialization_checkpoint=str(Path(student_checkpoint).resolve()),
                   student_initialization_sha256=file_sha256(student_checkpoint))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', cfg)
    print(output/'config.json', flush=True)


def check_config(cfg):
    for key in ('optimizer_updates', 'batch_size', 'grad_acc_steps', 'expected_world_size', 'save_every', 'val_every', 'log_every'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= cfg['warmup_steps'] < cfg['optimizer_updates']:
        raise ValueError('Warmup must finish before the final update')
    weights = [cfg['loss_weight_2d'], cfg['loss_weight_3d']]
    if any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) == 0:
        raise ValueError('Loss weights must be finite, nonnegative and not both zero')
    if not math.isfinite(cfg['learning_rate']) or cfg['learning_rate'] <= 0:
        raise ValueError('Learning rate must be positive and finite')
    if cfg['downsample_factor'] not in (1,4) or cfg['patch_size'] != [v//cfg['downsample_factor'] for v in (64,256,256)]:
        raise ValueError('Unsupported distillation geometry')
    sampling_from_config(cfg)
    if cfg['data_config'].get('paired_native') and cfg['downsample_factor'] != 4:
        raise ValueError('Paired native CT requires the 9.6um student')
    if recipe:=cfg.get('ema_batchnorm_calibration'):
        if (recipe.get('method')!='training_only_cumulative' or recipe.get('draw_start')!=0
                or recipe.get('global_samples',0)<1 or recipe.get('batch_size_per_gpu',0)<1
                or recipe['global_samples']%(recipe['batch_size_per_gpu']*cfg['expected_world_size'])):
            raise ValueError('Invalid deterministic training-only EMA calibration recipe')
    if cfg.get('student_initialization_checkpoint'):
        if file_sha256(cfg['student_initialization_checkpoint'])!=cfg['student_initialization_sha256']:
            raise ValueError('Changed student checkpoint')
    if recipe:=cfg['data_config'].get('paired_native'):
        if file_sha256(recipe['manifest'])!=recipe['sha256']:raise ValueError('Changed native pairing manifest')
        if recipe.get('quality_file') and file_sha256(recipe['quality_file'])!=recipe['quality_sha256']:
            raise ValueError('Changed native eligibility scores')
    records=json.loads(Path(cfg['data_config']['manifest']).read_text())['segments']
    for scroll in {r['scroll'] for r in records}:
        cutoff=cfg['background_thresholds'].get(scroll)
        if cutoff is None or not math.isfinite(cutoff) or not 0 <= cutoff <= 255:
            raise ValueError(f'Missing or invalid background cutoff: {scroll}')
    calibrated=cfg['data_config'].get('distillation', {}).get('coarse', {})
    if calibrated and cfg['background_thresholds'] != calibrated['background_thresholds']:
        raise ValueError('Background thresholds differ from the recorded calibration')
    if cfg.get('background_calibration_sha256'):
        calibration=Path(cfg['data_config']['background_calibration_dir'])/'calibration.json'
        if file_sha256(calibration)!=cfg['background_calibration_sha256']:
            raise ValueError('Background calibration changed')
    for key, path in (('teacher_sha256', cfg['teacher_checkpoint']),
                      ('manifest_sha256', cfg['data_config']['manifest']),
                      ('patches_sha256', Path(cfg['data_config']['manifest']).parent/'patches.npz')):
        if file_sha256(path) != cfg[key]:
            raise ValueError(f'Changed input: {key}')


def preview_ink_boundary(labels, support):
    """Ink pixels touching a supported non-ink pixel; ignore mask/image edges."""
    ink=labels.bool() & support.bool()
    background=~labels.bool() & support.bool()
    boundary=torch.zeros_like(ink)
    boundary[...,1:,:] |= ink[...,1:,:] & background[...,:-1,:]
    boundary[...,:-1,:] |= ink[...,:-1,:] & background[...,1:,:]
    boundary[...,:,1:] |= ink[...,:,1:] & background[...,:,:-1]
    boundary[...,:,:-1] |= ink[...,:,:-1] & background[...,:,1:]
    return boundary


def preview_priority(labels, support, draw_id):
    """Prefer supported ink boundaries, then mixed labels; fixed across models."""
    positives=int((labels.bool() & support.bool()).sum())
    negatives=int((~labels.bool() & support.bool()).sum())
    boundaries=int(preview_ink_boundary(labels,support).sum())
    return positives>0,boundaries,min(positives,negatives),-int(draw_id)


def preview_section_yx(labels, support, shape):
    """Choose intersecting XZ/YZ cuts through a supported human-positive pixel."""
    ink=np.asarray(labels,dtype=bool)&np.asarray(support,dtype=bool)
    boundary=preview_ink_boundary(torch.as_tensor(np.asarray(labels)),
                                  torch.as_tensor(np.asarray(support))).numpy()
    if boundary.any():ink=boundary
    while ink.ndim>2:ink=ink[0]
    if ink.any():
        # Tie breaks prefer the center and then the smaller coordinate.
        rows=ink.sum(1);ys=np.flatnonzero(rows==rows.max())
        y=int(min(ys,key=lambda v:(abs(v-(ink.shape[0]-1)/2),v)))
        xs=np.flatnonzero(ink[y]);x=int(min(xs,key=lambda v:(abs(v-(ink.shape[1]-1)/2),v)))
    else:y,x=(n//2 for n in ink.shape)
    return tuple(min(n-1,int((v+.5)*n/original)) for v,n,original in zip((y,x),shape,ink.shape))


def preview(path, raw_low, output, target, labels, support):
    from PIL import Image, ImageDraw
    ct, pred, truth = raw_low[0,0].cpu().numpy(), output['ink_3d_logits'][0,0].sigmoid().cpu().numpy(), target['volume'][0,0].cpu().numpy()
    y,x=preview_section_yx(labels.cpu().numpy(),support.cpu().numpy(),ct.shape[-2:])
    profile=truth[:,y,x];z=int(profile.argmax()) if profile.max()>0 else ct.shape[0]//2
    rows=[]
    for name, arrays in [(f'XY z={z}', [a[z] for a in (ct,truth,pred)]),
                         (f'XZ y={y}', [a[:,y,:] for a in (ct,truth,pred)]),
                         (f'YZ x={x}', [a[:,:,x] for a in (ct,truth,pred)])]:
        rows.append([(f'{name} {title}', a) for title,a in zip(('CT','teacher 3D','student 3D'),arrays)])
    rows.append([('human labels (evaluation only)', labels[0,0].cpu().numpy()),
                 ('teacher 2D', target['projection'][0,0].cpu().numpy()),
                 ('student 2D', output['ink'][0,0].sigmoid().cpu().numpy())])
    rows.append([('human valid mask',support[0,0].cpu().numpy()),
                 ('3D target valid fraction',target['volume_weight'][0,0,0].cpu().numpy()),
                 ('absolute 2D difference',(output['ink'][0,0].sigmoid()-target['projection'][0,0]).abs().cpu().numpy())])
    canvas=Image.new('RGB',(768,5*152),'black'); draw=ImageDraw.Draw(canvas)
    for r,row in enumerate(rows):
        for c,(title,a) in enumerate(row):
            pic=Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8)).resize((256,128))
            canvas.paste(pic,(c*256,r*152+24)); draw.text((c*256+2,r*152+3),title,fill='white')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,compress_level=9)


def calibration_loader(dataset,cfg,accelerator):
    recipe=cfg['ema_batchnorm_calibration']
    sampler=RankDrawSampler(0,recipe['global_samples'],accelerator.process_index,accelerator.num_processes)
    options=dict(batch_size=recipe['batch_size_per_gpu'],num_workers=cfg['dataloader_workers'],
                 pin_memory=True,worker_init_fn=seed_worker,
                 generator=torch.Generator().manual_seed(cfg['seed']+9000+accelerator.process_index))
    if options['num_workers']:options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
    return DataLoader(dataset,sampler=sampler,**options)


def calibrate_ema(ema,loader,cfg,accelerator,step):
    started=time.monotonic()
    result=recalibrate_batchnorm(ema,loader,
        lambda model,batch:model.forward_3d(batch_input(move_batch(batch,accelerator.device),cfg['downsample_factor'])),
        autocast=accelerator.autocast)
    if accelerator.num_processes>1:
        hashes=[None]*accelerator.num_processes
        torch.distributed.all_gather_object(hashes,result['buffer_sha256'])
        if len(set(hashes))!=1:raise ValueError('Calibrated EMA buffers differ between ranks')
    return {**result,'method':'training_only_cumulative','optimizer_update':step,
            'recipe':cfg['ema_batchnorm_calibration'],'manifest_sha256':cfg['manifest_sha256'],
            'patches_sha256':cfg['patches_sha256'],'seconds':time.monotonic()-started}


@torch.no_grad()
def validate(ema, teacher, loader, dataset, accelerator, cfg, out, step, run, prefix='validation'):
    ema.eval()
    scrolls=dataset.base.scroll_vocabulary
    sums=torch.zeros((len(scrolls),12),dtype=torch.float64,device=accelerator.device)
    candidates={}
    rejected=torch.zeros(len(scrolls),device=accelerator.device,dtype=torch.long)
    for batch in loader:
        batch=move_batch(batch,accelerator.device)
        if 'paired_available' in batch and prefix!='validation_fine':
            keep=~batch['paired_available'].bool() | batch['native_eligible'].bool()
            for rid in batch['record_id'][~keep].tolist():
                rejected[scrolls.index(dataset.base.records[rid]['scroll'])]+=1
            if not bool(keep.any()):continue
            batch={k:v[keep] for k,v in batch.items()}
        low=batch_input(batch, cfg['downsample_factor'])
        with accelerator.autocast():
            targets=teacher_targets(teacher(batch['teacher_image'],batch['valid_3d']),batch['valid_3d'],cfg['downsample_factor'],
                                    foreground_mask(batch,dataset.base.records,cfg['background_thresholds']))
            targets=mask_pair_targets(targets,batch)
            prediction=ema(low,targets['volume_weight']>0)
        _,losses=distillation_loss(prediction,targets,cfg['loss_weight_2d'],cfg['loss_weight_3d'])
        mae3=weighted_mean_per_sample((prediction['ink_3d_logits'].sigmoid()-targets['volume']).abs(),targets['volume_weight'])
        mae2=weighted_mean_per_sample((prediction['ink'].sigmoid()-targets['projection']).abs(),targets['projection_weight'])
        p=F.interpolate(prediction['ink'].sigmoid(),size=(256,256),mode='bilinear',align_corners=False)
        t=F.interpolate(targets['projection'],size=(256,256),mode='bilinear',align_corners=False)
        for i,record_id in enumerate(batch['record_id'].tolist()):
            scroll=dataset.base.records[record_id]['scroll']; row=sums[scrolls.index(scroll)]
            row[:5]+=torch.stack((losses['bce_2d'][i],losses['bce_3d'][i],mae2[i],mae3[i],p.new_tensor(1)))
            y=batch['labels_2d'][i].bool(); m=batch['mask_2d'][i].bool()
            for offset,prob in ((5,p[i]),(8,t[i])):
                yes=prob>=.5
                row[offset:offset+3]+=torch.stack(((yes&y&m).sum(),(yes&~y&m).sum(),(~yes&y&m).sum()))
            row[11]+=m.sum()
            draw_id=int(batch['draw_id'][i]);priority=preview_priority(y,m,draw_id)
            if (bool(batch.get('native_domain',torch.ones(len(batch['record_id']),device=accelerator.device))[i]) and bool((y&m).any())) and (scroll not in candidates or priority>candidates[scroll]['priority']):
                cpu=lambda value:value[i:i+1].detach().float().cpu()
                candidates[scroll]={'priority':priority,'draw_id':draw_id,'record_id':record_id,
                    'human_ink_pixels':int((y&m).sum()),
                    'human_boundary_pixels':int(preview_ink_boundary(y,m).sum()),
                    'segment':dataset.base.records[record_id]['segment'],
                    'yx':batch['yx'][i].cpu().tolist(),
                    'raw':cpu(low),'prediction':{k:cpu(prediction[k]) for k in ('ink','ink_3d_logits')},
                    'target':{k:cpu(targets[k]) for k in ('volume','projection','volume_weight')},
                    'labels':cpu(batch['labels_2d']),'support':cpu(batch['mask_2d'])}
    sums=accelerator.reduce(sums,reduction='sum').cpu()
    rejected=accelerator.reduce(rejected,reduction='sum').cpu()
    directory=out/('previews' if prefix=='validation' else prefix+'_previews')
    local={}
    for scroll,candidate in candidates.items():
        path=directory/f'{step:06d}_{scroll}_rank{accelerator.process_index}.png'
        preview(path,candidate['raw'],candidate['prediction'],candidate['target'],candidate['labels'],candidate['support'])
        local[scroll]={k:candidate[k] for k in ('priority','draw_id','record_id','segment','yx','human_ink_pixels','human_boundary_pixels')}
        local[scroll]['path']=str(path)
    gathered=[local]
    if accelerator.num_processes>1:
        import torch.distributed as dist
        gathered=[None]*accelerator.num_processes;dist.all_gather_object(gathered,local)
    if accelerator.is_main_process:
        import shutil
        selection={}
        for scroll in scrolls:
            choices=[rank[scroll] for rank in gathered if scroll in rank]
            if not choices:continue
            chosen=max(choices,key=lambda item:tuple(item['priority']))
            path=directory/f'{step:06d}_{scroll}.png';shutil.copyfile(chosen['path'],path)
            selection[scroll]={**chosen,'path':str(path)}
            if run:
                import wandb
                run.log({f'{prefix}_images/{scroll}':wandb.Image(str(path),caption=
                    f"{prefix.upper()} (training_preview is visualization only, not validation). Native CT, human-ink-boundary-selected fixed draw {chosen['draw_id']}; "
                    f"{chosen['human_ink_pixels']} ink pixels, {chosen['human_boundary_pixels']} supported boundary pixels. "
                    "Native renders only, with supported human ink. CT cross-sections pass through a boundary when available. Full-set metrics include negatives.")},step=step)
        write_json(out/f'{prefix}_preview_selection_{step:06d}.json',selection)
    del candidates
    if prefix=='training_preview':return {}
    metrics={f'{prefix}/{scroll}/native_alignment_rejected':int(rejected[i]) for i,scroll in enumerate(scrolls)}
    for scroll,row in zip(scrolls,sums):
        if not row[4]:continue
        for name,value in zip(('bce_2d','bce_3d','mae_2d','mae_3d'),row[:4]/row[4]):
            metrics[f'{prefix}/{scroll}/{name}']=float(value)
        for name,offset in (('student',5),('downsampled_teacher',8)):
            tp,fp,fn=row[offset:offset+3]
            metrics[f'{prefix}/{scroll}/{name}_human_dice']=float(2*tp/(2*tp+fp+fn).clamp_min(1))
        metrics[f'{prefix}/{scroll}/human_valid_pixels']=int(row[11])
    for name in ('bce_2d','bce_3d','mae_2d','mae_3d','student_human_dice','downsampled_teacher_human_dice'):
        values=[value for key,value in metrics.items() if key.endswith('/'+name)]
        if values:metrics[f'{prefix}/macro/{name}']=sum(values)/len(values)
    if accelerator.is_main_process:
        write_json(out/f'{prefix}_{step:06d}.json',metrics)
        if run:run.log(metrics,step=step)
    return metrics


def train(config_path, *, smoke=False, resume=None):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    cfg=json.loads(Path(config_path).read_text()); check_config(cfg)
    if smoke:
        cfg.update(optimizer_updates=2,warmup_steps=1,val_every=2,save_every=1,
                   expected_world_size=int(os.environ.get('WORLD_SIZE','1')),wandb_mode='disabled',
                   out_dir=str(Path(cfg['out_dir']).parent/'smoke'),dataloader_workers=0,log_every=1)
        cfg['data_config']['val_patches_per_scroll']=2
        check_config(cfg)
    acc=Accelerator(mixed_precision=cfg['mixed_precision'])
    if acc.num_processes!=cfg['expected_world_size']:
        raise ValueError('World size differs from configured effective batch')
    set_seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    out=Path(cfg['out_dir'])
    if acc.is_main_process:
        if out.exists() and not resume:raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'resolved_config.json',cfg)
    acc.wait_for_everyone()
    teacher, teacher_config=load_hecate(cfg['teacher_checkpoint'],acc.device)
    if teacher_config['image_normalization'] != cfg['teacher_normalization']:
        raise ValueError('Teacher preprocessing differs from its checkpoint')
    config_for_model=deepcopy(cfg)
    config_for_model['model_config']['canonical']['source_dir']=str(canonical_source_dir())
    model=make_model(InkConfig.from_mapping(config_for_model))
    if cfg.get('student_initialization_checkpoint') and not resume:
        initial, initial_config=load_hecate(cfg['student_initialization_checkpoint'])
        if sampling_from_config(initial_config) != sampling_from_config(cfg):
            raise ValueError('Student initialization architecture mismatch')
        load_model_state(model,initial.state_dict())
        del initial
        print('INITIALIZED_STUDENT_EMA fresh_optimizer=True',flush=True)
    if cfg.get('sync_batchnorm',True) and acc.num_processes>1:
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ema=deepcopy(model).eval().requires_grad_(False).to(acc.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:schedule_scale(step,cfg['warmup_steps'],cfg['optimizer_updates']))
    step=0; restored_rng=None; prior_run=None; batch_transition=False
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        batch_transition=validate_resume_recipe(state['config'],cfg,resume,state['optimizer_step'])
        load_model_state(model,state['model']);load_model_state(ema,state['ema_model'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['lr_scheduler'])
        step=state['optimizer_step'];restored_rng=state['rng_by_rank'][acc.process_index];prior_run=state.get('wandb_run_id')
        del state
    model,optimizer=acc.prepare(model,optimizer)
    data=ResolutionDataset(cfg['data_config'],cfg['teacher_normalization'])
    val=ResolutionDataset(cfg['data_config'],cfg['teacher_normalization'],validation=True)
    effective=cfg['batch_size']*cfg['grad_acc_steps']*acc.num_processes
    sampler=RankDrawSampler(step*effective,cfg['optimizer_updates']*effective,acc.process_index,acc.num_processes)
    val_sampler=RankDrawSampler(0,len(val),acc.process_index,acc.num_processes)
    options=dict(batch_size=cfg['batch_size'],num_workers=cfg['dataloader_workers'],pin_memory=True,
                 worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(cfg['seed']+acc.process_index))
    if options['num_workers']:options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(data,sampler=sampler,**options); val_loader=DataLoader(val,sampler=val_sampler,**options)
    fine_val=fine_val_loader=None
    if cfg.get('log_fine_validation'):
        fine_config=deepcopy(cfg['data_config'])
        fine_config['paired_native']['force_fine_validation']=True
        fine_val=ResolutionDataset(fine_config,cfg['teacher_normalization'],validation=True)
        fine_val_loader=DataLoader(fine_val,sampler=val_sampler,**options)
    training_preview=training_preview_loader=None
    if cfg.get('training_preview_scroll'):
        preview_config=deepcopy(cfg['data_config'])
        preview_config['training_preview_scroll']=cfg['training_preview_scroll']
        training_preview=ResolutionDataset(preview_config,cfg['teacher_normalization'])
        training_preview_loader=DataLoader(training_preview,
            sampler=RankDrawSampler(0,len(training_preview),acc.process_index,acc.num_processes),**options)
    calibration=calibration_loader(data,cfg,acc) if cfg.get('ema_batchnorm_calibration') else None
    if restored_rng:restore_rng(restored_rng)
    else:set_seed(cfg['seed']+acc.process_index)
    run=None
    if acc.is_main_process and cfg['wandb_mode']!='disabled':
        import wandb
        kwargs={'id':prior_run,'resume':'must'} if prior_run else {}
        run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],name=cfg['wandb_run_name'],
                       config=None if batch_transition else cfg,dir=str(out),mode=cfg['wandb_mode'],**kwargs)
        if batch_transition:
            run.config.update(cfg,allow_val_change=True)
            # A checkpoint's validation step may already be committed in W&B.
            # Record the event without trying to append to that historical step.
            if cfg.get('loss_transitions'):
                run.summary['latest_loss_transition']=cfg['loss_transitions'][-1]
            elif cfg.get('data_transitions'):
                run.summary['latest_data_transition']=cfg['data_transitions'][-1]
            else:
                run.summary['latest_batch_transition']=cfg['batch_transitions'][-1]
        artifact=wandb.Artifact('native-resolution-distillation-recipe',type='run-inputs')
        artifact.add_file(str(out/'resolved_config.json'))
        artifact.add_file(cfg['data_config']['manifest'])
        if cfg['data_config'].get('paired_native'):artifact.add_file(cfg['data_config']['paired_native']['manifest'])
        artifact.add_file(str(Path(cfg['data_config']['manifest']).parent/'patches.npz'))
        if cfg['data_config'].get('paired_native',{}).get('quality_file'):
            artifact.add_file(cfg['data_config']['paired_native']['quality_file'])
        add_code_to_artifact(artifact)
        run.log_artifact(artifact)
        print('WANDB_URL='+run.url,flush=True)
    stopping=[False]
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stopping.__setitem__(0,True))
    calibrated=None

    def ensure_calibrated():
        nonlocal calibrated
        if calibration is not None and (calibrated is None or calibrated['optimizer_update']!=step):
            calibrated=calibrate_ema(ema,calibration,cfg,acc,step)
            if acc.is_main_process:
                write_json(out/f'ema_calibration_{step:06d}.json',calibrated)
                print('EMA_BN_CALIBRATED '+json.dumps(calibrated),flush=True)
                if run:run.log({'ema_calibration/seconds':calibrated['seconds'],
                                 'ema_calibration/training_samples':cfg['ema_batchnorm_calibration']['global_samples']},step=step)

    def evaluate():
        ensure_calibrated()
        validate(ema,teacher,val_loader,val,acc,cfg,out,step,run)
        if fine_val_loader is not None:validate(ema,teacher,fine_val_loader,fine_val,acc,cfg,out,step,run,prefix='validation_fine')
        if training_preview_loader is not None:validate(ema,teacher,training_preview_loader,training_preview,acc,cfg,out,step,run,prefix='training_preview')
        if cfg.get('log_live_validation'):
            live=acc.unwrap_model(model);was_training=live.training
            try:validate(live,teacher,val_loader,val,acc,cfg,out,step,run,prefix='validation_live')
            finally:live.train(was_training)

    def save():
        ensure_calibrated()
        local=rng_state(); states=[local]
        if acc.num_processes>1:
            import torch.distributed as dist
            states=[None]*acc.num_processes;dist.all_gather_object(states,local)
        if acc.is_main_process:
            dest=out/f'ckpt_{step:06d}.pth';temp=dest.with_suffix('.partial')
            torch.save({'model':acc.unwrap_model(model).state_dict(),'ema_model':ema.state_dict(),
                        'optimizer':optimizer.state_dict(),'lr_scheduler':scheduler.state_dict(),
                        'optimizer_step':step,'config':cfg,'rng_by_rank':states,
                        'wandb_run_id':run.id if run else None,
                        'ema_batchnorm_calibration':calibrated},temp)
            os.replace(temp,dest)
            alias=out/'latest.partial';alias.symlink_to(dest.name);os.replace(alias,out/'latest.pth')
            write_json(out/'status.json',{'optimizer_update':step,'checkpoint':str(dest),
                       'finished':step==cfg['optimizer_updates'],'production':not smoke})
        acc.wait_for_everyone()

    evaluate()
    if resume and calibration is not None and not (out/f'ckpt_{step:06d}.pth').exists():save()
    model.train();optimizer.zero_grad(set_to_none=True)
    started=time.monotonic(); metrics=torch.zeros(3,device=acc.device)
    for micro,batch in enumerate(loader):
        batch=move_batch(batch,acc.device);low=batch_input(batch, cfg['downsample_factor'])
        with torch.no_grad(),acc.autocast():
            targets=teacher_targets(teacher(batch['teacher_image'],batch['valid_3d']),batch['valid_3d'],cfg['downsample_factor'],
                                    foreground_mask(batch,data.base.records,cfg['background_thresholds']))
        targets=mask_pair_targets(targets,batch)
        final_micro=(micro+1)%cfg['grad_acc_steps']==0
        from contextlib import nullcontext
        with (nullcontext() if final_micro else acc.no_sync(model)):
            with acc.autocast():
                prediction=model(low,targets['volume_weight']>0)
                loss,parts=distillation_loss(prediction,targets,cfg['loss_weight_2d'],cfg['loss_weight_3d'])
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite distillation loss')
            acc.backward(loss/cfg['grad_acc_steps'])
        metrics+=torch.stack((loss.detach(),parts['bce_2d'].mean(),parts['bce_3d'].mean()))/cfg['grad_acc_steps']
        if not final_micro:continue
        grad=acc.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
        if not torch.isfinite(grad):raise FloatingPointError('Non-finite gradient norm')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        scheduler.step();step+=1
        update_ema(ema,acc.unwrap_model(model),cfg['ema_decay'])
        values=acc.reduce(metrics,reduction='mean');metrics.zero_()
        if step%cfg['log_every']==0 or step==1:
            report=dict(zip(('train/loss','train/bce_2d','train/bce_3d'),map(float,values)))
            report.update({'train/lr_next':optimizer.param_groups[0]['lr'],'train/gradient_norm':float(grad),
                           'train/batch_per_gpu':cfg['batch_size'],'train/gradient_accumulation':cfg['grad_acc_steps'],
                           'train/effective_batch':effective})
            if 'native_eligible' in batch:
                report['train/native_eligible_last_microbatch']=float(acc.reduce(batch['native_eligible'].float().mean(),reduction='mean'))
            if 'native_domain' in batch:
                report.update({'train/native_fraction_last_microbatch':float(acc.reduce(batch['native_domain'].mean(),reduction='mean')),
                               'train/contrast_last_microbatch':float(acc.reduce(batch['contrast_gain'].mean(),reduction='mean')),
                               'train/blur_sigma_last_microbatch':float(acc.reduce(batch['blur_sigma'].mean(),reduction='mean'))})
            if acc.is_main_process:
                print(f'update={step}/{cfg["optimizer_updates"]} '+json.dumps(report),flush=True)
                if run:run.log(report,step=step)
        stop=bool(acc.reduce(torch.tensor(int(stopping[0]),device=acc.device),reduction='sum'))
        if step%cfg['val_every']==0:
            evaluate()
        if step%cfg['save_every']==0 or stop or step==cfg['optimizer_updates']:save()
        if stop:break
    if acc.is_main_process:
        write_json(out/'completion.json',{'optimizer_update':step,'production':not smoke,
                   'elapsed_seconds_including_periodic_validation':time.monotonic()-started})
        if run:run.finish()
    acc.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--factor',type=int,choices=(1,4),default=4);p.add_argument('--data-config');p.add_argument('--student-checkpoint')
    p=sub.add_parser('export');p.add_argument('checkpoint');p.add_argument('output')
    p=sub.add_parser('smoke');p.add_argument('config');p.add_argument('--resume')
    p=sub.add_parser('train');p.add_argument('config');p.add_argument('--production',action='store_true');p.add_argument('--resume')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.checkpoint,args.output,args.factor,args.data_config,args.student_checkpoint)
    elif args.command=='export':print(export_ema(args.checkpoint,args.output))
    elif args.command=='smoke':train(args.config,smoke=True,resume=args.resume)
    elif not args.production:parser.error('Production training requires --production; preparation never launches it')
    else:train(args.config,resume=args.resume)


if __name__=='__main__':main()
