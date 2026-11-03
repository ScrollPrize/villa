"""Actual-data first-order audit, run separately from timing pilots."""
import torch
from .fiber_grouped_gather import width_gather_mode
from . import fiber_soft_winding as soft


def audit(state,raw,widths,data,nodes,config,pair_cost,extension):
    edges=torch.arange(min(257,len(data['i'])),device=state.device).repeat(2)
    params=(state,raw,extension.values)
    original=dict(config,reuse_constraint_work=False,bucket_uv_direct_meetings=False,
                  bucket_uv_contiguous=False,width_gather_mode='original')
    saved=extension.config
    def evaluate(cfg):
        extension.config=cfg
        mode=cfg['width_gather_mode']
        cost=pair_cost
        if cfg.get('width_gather_scope')=='pairs':
            def cost(*args,**kwargs):
                with width_gather_mode(mode):return pair_cost(*args,**kwargs)
        with width_gather_mode('original' if cfg.get('width_gather_scope')=='pairs' else mode):
            loss,terms=soft.objective(state,raw,widths,data,nodes,cfg,edges,cost,extension)
            grad=torch.autograd.grad(loss,params)
        return terms,grad
    try:
        reference,baseline=evaluate(original)
        rows=[]
        for mode,contiguous,scope in [('original',False,'all'),('select',False,'all'),('grouped',False,'all'),('select',True,'all'),('select',False,'pairs'),('grouped',False,'pairs')]:
            cfg=dict(original,reuse_constraint_work=True,bucket_uv_direct_meetings=True,
                     width_gather_mode=mode,bucket_uv_contiguous=contiguous,width_gather_scope=scope)
            terms,grads=evaluate(cfg)
            absdiff=[float((a-b).abs().max()) for a,b in zip(baseline,grads)]
            relative=[float((a-b).abs().max()/a.abs().max().clamp_min(1e-20)) for a,b in zip(baseline,grads)]
            for key in reference:torch.testing.assert_close(terms[key],reference[key],rtol=1e-5,atol=1e-6)
            rows.append(dict(mode=mode,contiguous=contiguous,scope=scope,
                passed=max(relative)<=1e-4,
                term_abs={k:float((terms[k]-reference[k]).abs().detach()) for k in reference},
                gradient_reference_max=[float(g.abs().max()) for g in baseline],
                gradient_candidate_max=[float(g.abs().max()) for g in grads],
                gradient_max_abs=absdiff,gradient_relative_to_max=relative))
        # Isolate pair/unary gradients from deformation gradients at the same
        # state. This is diagnostic, never used to change the training objective.
        isolated=[]
        for include_extension in (False,True):
            gradients=[]
            for mode in ('original','select','grouped'):
                cfg=dict(original,width_gather_mode=mode)
                extension.config=cfg
                with width_gather_mode(mode):
                    loss,_=soft.objective(state,raw,widths,data,nodes,cfg,edges,pair_cost,
                                         extension if include_extension else None)
                    g=torch.autograd.grad(loss,raw)[0]
                gradients.append(g)
            isolated.append(dict(extension=include_extension,
                gradient_max=[float(g.abs().max()) for g in gradients],
                max_abs_vs_original=[float((g-gradients[0]).abs().max()) for g in gradients]))
        return dict(events=len(edges),parameters=['state','raw_widths','absolute_uv'],rows=rows,isolated=isolated)
    finally:extension.config=saved
