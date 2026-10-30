"""Read-only H/V length density, with optional disabled-neighbor placement audit."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from . import fiber_uv_layout as uv
from .fiber_uv_kollesis import Maps, all_coordinates, segment_cells, ratio
from .fiber_uv_hv_rejection import save_figure


def refit_disabled(d, state, widths, cfg, batch=4096):
    """Frozen enabled MAP neighbors; canonical pair costs, no unary/disabled peers.

    Evaluate Kx1 or 1xK instead of KxK. Original endpoint orientation, N/E
    normalization and whole-graph degree calibration are retained. Only diagnostic
    choices are returned; inputs and fitted logits/geometry remain unchanged.
    """
    import torch
    torch.set_num_threads(8)
    n=len(state); k=len(widths); e=len(d['i'])
    p=softmax(state[:,:k+1].astype(float),axis=1)
    disabled=p[:,-1]>.5; pick=p[:,:k].argmax(1)
    costs=torch.zeros((n,k),dtype=torch.float64)
    violations=torch.zeros((n,k),dtype=torch.int32)
    support=np.zeros(n,np.int64)
    t=torch.as_tensor(state,dtype=torch.float64); w=torch.as_tensor(widths,dtype=torch.float64)
    bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
    fields={key:torch.as_tensor(d[key]) for key in ('family','phase','z','degree')}
    degree=fields['degree'].double(); mean_degree=degree.mean()
    table=torch.as_tensor(uv.make_distance_table(float(np.max(d['distance'],initial=0)),cfg.get('distance_sigma',.5)))
    chosen=torch.as_tensor(pick)
    scale=n/max(e,1); start=time.monotonic()
    def coords(ids,ds):
        return uv.coordinates(t[ids],fields['phase'][ids],fields['z'][ids],fields['family'][ids],
                              torch.as_tensor(ds,dtype=torch.float64),w,bounds)
    # Process each orientation independently: never reverse the measured sign.
    with torch.no_grad():
        for side in (0,1):
            a=d['i']; b=d['j']
            selected=np.flatnonzero((disabled[a]&~disabled[b]) if side==0 else (~disabled[a]&disabled[b]))
            for first in range(0,len(selected),batch):
                ix=selected[first:first+batch]; i=torch.as_tensor(a[ix]); j=torch.as_tensor(b[ix])
                _,va,qa=coords(i,d['ds_a'][ix]); _,vb,qb=coords(j,d['ds_b'][ix])
                if side==0:
                    qb=qb.gather(1,chosen[j,None]); vb=vb.gather(1,chosen[j,None]); owner=i
                else:
                    qa=qa.gather(1,chosen[i,None]); va=va.gather(1,chosen[i,None]); owner=j
                args=[torch.as_tensor(d[key][ix]) for key in ('phase_delta','sign','distance')]
                cost,order,_,_=uv.pair_cost_components(qa,qb,va,vb,args[0],fields['family'][j]-fields['family'][i],
                    args[1],args[2],table,cfg)
                charge=uv.calibrated_sign_charge(degree[i],degree[j],mean_degree,max(e,1),n,
                    cfg.get('rejection_cost',10.),cfg.get('sign_charge_multiplier',4.))
                cost=(cost+charge[:,None,None]*(order<0)).squeeze(2 if side==0 else 1)*scale
                costs.index_add_(0,owner,cost)
                violations.index_add_(0,owner,(order<0).squeeze(2 if side==0 else 1).int())
                np.add.at(support,owner.numpy(),1)
                if first%(batch*64)==0:
                    print(f'density placement endpoint{side}: {first+len(ix):,}/{len(selected):,} edges, {time.monotonic()-start:.1f}s',flush=True)
        links=d['links']
        for first in range(0,len(links),batch):
            link=links[first:first+batch]; ai=link[:,0].astype(int); bj=link[:,1].astype(int)
            for side in (0,1):
                ok=(disabled[ai]&~disabled[bj]) if side==0 else (~disabled[ai]&disabled[bj])
                if not ok.any():continue
                vlink=link[ok]; i=torch.as_tensor(ai[ok]); j=torch.as_tensor(bj[ok])
                ua,va,_=coords(i,vlink[:,2]); ub,vb,_=coords(j,vlink[:,3])
                if side==0:
                    ub=ub.gather(1,chosen[j,None]);vb=vb.gather(1,chosen[j,None]);owner=i
                else:
                    ua=ua.gather(1,chosen[i,None]);va=va.gather(1,chosen[i,None]);owner=j
                cost=(((ua[:,:,None]-ub[:,None,:])/cfg.get('continuation_scale',128.)).square()+
                      ((va[:,:,None]-vb[:,None,:])/cfg.get('continuation_scale',128.)).square())
                weight=cfg.get('continuation_weight',10.)
                costs.index_add_(0,owner,cost.squeeze(2 if side==0 else 1)*scale*weight)
                if weight>0:np.add.at(support,owner.numpy(),1)
    costs=costs.numpy();violations=violations.numpy();supported=disabled&(support>0)
    minimum=costs.min(1); replacement=costs.argmin(1)
    # Preserve previous placement if it is exactly tied for minimum.
    tied=costs[np.arange(n),pick]==minimum
    replacement[tied]=pick[tied]
    updated=np.where(supported,replacement,pick)
    gaps=np.partition(costs,1,axis=1)[:,1]-minimum if k>1 else np.zeros(n)
    confidence=np.full(n,np.nan)
    confidence[supported]=softmax(-costs[supported]/cfg.get('target_temperature',.25),axis=1).max(1)
    oldbad=violations[np.arange(n),pick];newbad=violations[np.arange(n),updated]
    diagnostics=dict(disabled_pieces=int(disabled.sum()),supported_disabled=int(supported.sum()),
        unsupported_fallback=int((disabled&~supported).sum()),changed_disabled=int((updated!=pick).sum()),
        tied_minimum_disabled=int((supported&(gaps==0)).sum()),
        supported_cost_gap_quantiles=np.quantile(gaps[supported],[0,.25,.5,.75,.95,1]).tolist() if supported.any() else [],
        supported_cost_softmax_quantiles=np.quantile(confidence[supported],[0,.25,.5,.75,.95,1]).tolist() if supported.any() else [],
        sign_violations_before=int(oldbad[supported].sum()),sign_violations_after=int(newbad[supported].sum()),
        refit_seconds=time.monotonic()-start,
        interpretation='Diagnostic conditional minimum against enabled MAP neighbors, unit weights; no node/endpoint unary, no disabled peers, no geometry refit; cost-softmax is not calibrated probability')
    return updated,dict(support=support,cost_gap=gaps,cost_softmax=confidence,
        sign_violations_before=oldbad,sign_violations_after=newbad,costs=costs),diagnostics


def density_maps(d,u,v,enabled,choices,heat_step=3000.,profile_step=10000.):
    """Same UV rectangular area for every variant; exact source-length deposition."""
    endpoints=d['node_ds'][:,[0,2]]
    dx=(d['family']==0)[:,None]*endpoints;dy=(d['family']==1)[:,None]*endpoints
    selected={name:(u[np.arange(len(u)),pick,None],v[np.arange(len(v)),pick,None]) for name,pick in choices.items()}
    xmin=min(float((a+dx).min()) for a,b in selected.values()); xmax=max(float((a+dx).max()) for a,b in selected.values())
    ymin=min(float((b+dy).min()) for a,b in selected.values()); ymax=max(float((b+dy).max()) for a,b in selected.values())
    template=Maps((xmin,ymin,xmax,ymax),heat_step)
    # Snap common domain to complete heat cells, including a fixed-coordinate
    # fiber exactly on the maximum boundary. No tiny sliver-area density spikes.
    x0,y0=template.origin;x1=x0+template.shape[1]*heat_step;y1=y0+template.shape[0]*heat_step
    grids={};domain=[float(x0),float(y0),float(x1),float(y1)]
    for name,(a,b) in selected.items():
        grids[name]={}
        keep=enabled if name=='enabled' else np.ones(len(enabled),bool)
        for target,step in (('heat',heat_step),('profile',profile_step)):
            grid=Maps((xmin,ymin,xmax,ymax),step) if target=='heat' else Maps((x0,0,x1,0),step)
            xx=grid.origin[0]+np.arange(grid.shape[1])*step
            wx=np.maximum(0,np.minimum(xx+step,x1)-np.maximum(xx,x0))
            if target=='heat':area=np.full(grid.shape,step**2)
            else:area=(wx*(y1-y0))[None,:]
            for family in ('h','v'):
                for metric in ('length','count'):grid.values[family+'_'+metric]=np.zeros(grid.shape)
            for ids,x,y,length in segment_cells(grid,d,a,b,0,profile=target=='profile'):
                if not len(ids):continue
                family='h' if d['family'][ids[0]]==0 else 'v'
                ok=keep[ids]
                grid.add(family+'_length',x[ok],y[ok],length[ok]);grid.add(family+'_count',x[ok],y[ok],1.)
            grid.values['area']=area
            for family,code in (('h',0),('v',1)):
                total=d['lengths'][keep&(d['family']==code)].sum()
                if not np.isclose(grid.values[family+'_length'].sum(),total,rtol=1e-10,atol=1e-6):
                    raise ValueError('Fiber length not conserved')
            grids[name][target]=grid
    return grids,domain


def density(grid,family,metric):
    return 1e6*ratio(grid.values[family+'_'+metric],grid.values['area'])


def plot_density(grids,root,cfg,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root.mkdir(parents=True,exist_ok=True)
    names=list(grids);summaries={}
    for metric,title in (('length','Fiber length (base vx) / 1M base-vx²'),('count','Intersecting pieces / 1M base-vx²')):
        fig,axes=plt.subplots(2,2,figsize=(24,9),sharex=True,constrained_layout=True)
        columns={};summaries[metric]={}
        for row,family in enumerate(('h','v')):
            for name in names:
                grid=grids[name]['profile'];x=grid.origin[0]+(np.arange(grid.shape[1])+.5)*grid.step
                val=density(grid,family,metric)[0]
                columns['u_center']=x;columns[family+'_'+name]=val
                axes[row,0].plot(x,val,label=name,lw=.8)
                if name!='enabled':
                    delta=val-density(grids['enabled']['profile'],family,metric)[0]
                    axes[row,1].plot(x,delta,label=name+' minus enabled',lw=.8)
                ids=np.flatnonzero(np.isfinite(val));top=ids[np.argsort(val[ids])[-10:][::-1]]
                summaries[metric][family+'_'+name]=[dict(u=float(x[i]),density=float(val[i])) for i in top]
            for ax in axes[row]:ax.legend();ax.grid(alpha=.2);ax.set_ylabel(family.upper()+': '+title)
        axes[0,0].set_title('Absolute density');axes[0,1].set_title('Additional density from disabled pieces')
        for ax in axes[-1]:ax.set_xlabel('U (base vx)')
        fig.suptitle(f'{label}100 — inferred UV area density; {grid.step:g}vx U bins, common full V span')
        save_figure(fig,root,metric+'_profiles',cfg,label.lower())
        with (root/(metric+'_profiles.csv')).open('w') as f:
            writer=csv.writer(f);writer.writerow(columns);writer.writerows(zip(*columns.values()))
        values=[density(grids[name]['heat'],family,metric) for name in names for family in ('h','v')]
        positive=np.concatenate([a[np.isfinite(a)&(a>0)] for a in values])
        upper=max(1.,float(np.quantile(positive,.995))) if len(positive) else 1.
        heat_variants={name:grids[name]['heat'] for name in names}
        for name in names:
            if name=='enabled':continue
            base=grids[name]['heat'];extra=Maps((0,0,0,0),base.step)
            extra.origin=base.origin;extra.shape=base.shape
            extra.values={key:(val if key=='area' else val-grids['enabled']['heat'].values[key]) for key,val in base.values.items()}
            heat_variants[name+'_added_disabled']=extra
        for name,heat in heat_variants.items():
            panels=min(cfg.get('heatmap_panels',8),heat.shape[1])
            edges=np.linspace(0,heat.shape[1],panels+1).astype(int)
            fig,axes=plt.subplots(panels,2,figsize=(22,2.8*panels),squeeze=False,constrained_layout=True)
            for col,family in enumerate(('h','v')):
                val=density(heat,family,metric)
                for row,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
                    ax=axes[row,col]
                    im=ax.imshow(val[:,a:b],origin='lower',aspect='equal',interpolation='nearest',vmin=0,vmax=upper,
                        extent=(heat.origin[0]+a*heat.step,heat.origin[0]+b*heat.step,
                                heat.origin[1],heat.origin[1]+heat.shape[0]*heat.step))
                    if row==0:ax.set_title(family.upper()+' surface')
                    ax.set_ylabel('V (vx)');ax.set_xlabel('U (vx)')
                fig.colorbar(im,ax=list(axes[:,col]),label=title,shrink=.7)
            fig.suptitle(f'{label}100 — {name}; {heat.step:g}vx cells; same color scale across variants; >99.5th percentile clipped for display')
            save_figure(fig,root,name+'_'+metric+'_heatmaps',cfg,label.lower())
    for name,pair in grids.items():
        for target,g in pair.items():np.savez_compressed(root/(name+'_'+target+'.npz'),origin=g.origin,step=g.step,**g.values)
    uv.atomic_write(root/'profile_peaks.json',summaries,durable=False)


def run(cfg):
    from . import fiber_uv_kollesis,fiber_uv_hv_rejection
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True);reports=[]
    module_hashes={Path(module.__file__).name:uv.sha256_file(module.__file__)
        for module in (uv,fiber_uv_kollesis,fiber_uv_hv_rejection)}
    module_hashes[Path(__file__).name]=uv.sha256_file(__file__)
    for entry in cfg['solutions']:
        started=time.monotonic();source=Path(entry['run_root']);dest=root/entry['label'].lower();dest.mkdir(exist_ok=True)
        config=json.loads((source/'summary.json').read_text())['config']
        with np.load(config['tensor_cache']) as src:d={key:src[key] for key in src.files}
        with np.load(source/'solution.npz') as src:
            state=src['state'].copy();widths=src['widths'].astype(float)
            for key in ('source_id','start','end','family'):
                if not np.array_equal(src[key],d[key]):raise ValueError('Piece identity mismatch')
        if not uv.anchored(config):raise ValueError('Requires shared-anchor solution')
        p=softmax(state[:,:len(widths)+1].astype(float),axis=1);enabled=p[:,-1]<=.5;pick=p[:,:-1].argmax(1)
        print(f'density {entry["label"]}: disabled placement against enabled neighbors',flush=True)
        updated,details,refit=refit_disabled(d,state,widths,config,cfg.get('placement_batch',4096))
        np.savez_compressed(dest/'placement.npz',saved_choice=pick,refit_choice=updated,enabled=enabled,
            **{key:d[key] for key in ('source_id','start','end','family')},**details)
        del details
        u,v,_=all_coordinates(d,state,widths)
        grids,domain=density_maps(d,u,v,enabled,dict(enabled=pick,all_saved=pick,all_refit=updated),
            cfg.get('heatmap_step',3000.),cfg.get('profile_step',10000.))
        report=dict(label=entry['label'],run_root=str(source),config=cfg,domain=domain,placement=refit,
            families={family:dict(pieces=int((d['family']==code).sum()),
                enabled_pieces=int(((d['family']==code)&enabled).sum()),
                enabled_length=float(d['lengths'][(d['family']==code)&enabled].sum()),
                all_length=float(d['lengths'][d['family']==code].sum())) for family,code in (('h',0),('v',1))},
            solution_sha256=uv.sha256_file(source/'solution.npz'),tensor_sha256=uv.sha256_file(config['tensor_cache']),
            source_summary_sha256=uv.sha256_file(source/'summary.json'),module_hashes=module_hashes,
            implementation_sha256=uv.sha256_file(__file__),geometry_sha256=uv.sha256_file(uv.__file__),
            calculation_seconds=time.monotonic()-started)
        plot_density(grids,dest,cfg,entry['label']);report['total_seconds']=time.monotonic()-started
        uv.atomic_write(dest/'summary.json',report,durable=False);reports.append(report)
        print(json.dumps(report),flush=True)
    uv.atomic_write(root/'results.json',reports,durable=False)
    return reports
