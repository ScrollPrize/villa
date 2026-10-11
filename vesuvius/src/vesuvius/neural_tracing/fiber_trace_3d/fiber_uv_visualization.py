"""Read-only saved U/V solution plots and correctly aligned global cost curves."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from . import fiber_uv_layout as uv


def segments(d,state,widths,config):
    p,labels,keep,pick,u,v,_=uv.hard_geometry(d,state,widths,config)
    ids=np.flatnonzero(keep)
    ds=d['node_ds'][ids][:,[0,2]]
    x=u[ids,None]+(d['family'][ids]==0)[:,None]*ds
    y=v[ids,None]+(d['family'][ids]==1)[:,None]*ds
    joint=p[ids,pick[ids]]
    conditional=np.clip(joint/(1-p[ids,-1]),0,1)
    return ids,x,y,joint,conditional,p


def progression(history,final_energy,constant):
    """Logged pre-update energy at record s belongs to the preceding state."""
    rows=[]
    for n,h in enumerate(history):
        if n+1<len(history):
            if history[n+1]['step']!=h['step']+1:raise ValueError('Missing cost sweep')
            energy=history[n+1]['pre_update_energy']
        else:energy=final_energy
        rows.append(dict(sweep=h['step'],violations=h['sign_violations'],
            violation_percent=h['sign_violation_percent'],retained_length_percent=h['retained_arc_percent'],
            variable_expected_cost=energy,constant_geometry_cost=constant,global_expected_cost=energy+constant))
    return rows


class TiledRaster:
    """Record ordered drawing operations into tiles without a full-sized raster."""
    def __init__(self,width,height,tile=512):
        self.width=width;self.height=height;self.tile=tile;self.operations={}

    def add(self,kind,coords,kwargs,bounds):
        import math
        x0,y0,x1,y1=bounds;t=self.tile
        for ty in range(max(0,math.floor(y0/t)),min((self.height-1)//t,math.floor(y1/t))+1):
            for tx in range(max(0,math.floor(x0/t)),min((self.width-1)//t,math.floor(x1/t))+1):
                self.operations.setdefault((tx,ty),[]).append((kind,coords,kwargs))

    def line(self,coords,**kwargs):
        x0,y0,x1,y1=coords;r=kwargs.get('width',1)
        self.add('line',coords,kwargs,(min(x0,x1)-r,min(y0,y1)-r,max(x0,x1)+r,max(y0,y1)+r))

    def rectangle(self,coords,**kwargs):
        self.add('rectangle',coords,kwargs,coords)

    def text(self,coords,value,**kwargs):
        from PIL import ImageFont
        kwargs.pop('size',None)  # Raster labels retain the established bitmap font.
        b=ImageFont.load_default().getbbox(value);x,y=coords
        self.add('text',coords,dict(kwargs,text=value),(x+b[0],y+b[1],x+b[2],y+b[3]))

    def save(self,path):
        from PIL import Image,ImageDraw
        import tifffile
        t=self.tile
        def tiles():
            for y in range(0,self.height,t):
                for x in range(0,self.width,t):
                    image=Image.new('RGB',(t,t),'white');draw=ImageDraw.Draw(image,'RGBA')
                    for kind,coords,kwargs in self.operations.get((x//t,y//t),()):
                        shifted=tuple(v-(x if i%2==0 else y) for i,v in enumerate(coords))
                        getattr(draw,kind)(shifted,**kwargs)
                    yield np.asarray(image)
        tifffile.imwrite(path,tiles(),shape=(self.height,self.width,3),dtype=np.uint8,
                         photometric='rgb',tile=(t,t),compression='lzw',bigtiff=True,
                         maxworkers=8,metadata=None)


class VectorPDF:
    """Shared drawing sink: actual PDF paths/text, never a rasterized map."""
    def __init__(self,width,height):
        self.width=width;self.height=height;self.lines=[];self.colors=[];self.widths=[]
        self.rects=[];self.texts=[]

    def line(self,coords,fill,width=1):
        self.lines.append(np.asarray(coords).reshape(2,2));self.colors.append(np.array(fill)/255)
        self.widths.append(width)

    def rectangle(self,coords,fill):self.rects.append((coords,np.array(fill)/255))
    def text(self,coords,value,**kwargs):self.texts.append((coords,value,kwargs))

    def save(self,path):
        from matplotlib.figure import Figure
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        # Below the traditional PDF 200-inch page limit, with equal axis scale.
        scale=min(.75,12960/max(self.width,self.height))
        fig=Figure(figsize=(self.width*scale/72,self.height*scale/72))
        ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,self.width);ax.set_ylim(self.height,0);ax.set_axis_off()
        ax.add_collection(LineCollection(self.lines,colors=self.colors,
            linewidths=np.asarray(self.widths)*scale,capstyle='round',rasterized=False))
        for (x0,y0,x1,y1),color in self.rects:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=color,edgecolor='none'))
        for (x,y),value,kwargs in self.texts:
            ax.text(x,y,value,fontsize=kwargs.get('size',12)*scale,va='top',color='#222')
        fig.savefig(path,format='pdf');fig.clear()


class DrawingSinks:
    def __init__(self,*sinks):self.sinks=sinks
    def __getattr__(self,name):
        def forward(*args,**kwargs):
            for sink in self.sinks:getattr(sink,name)(*args,**kwargs)
        return forward


def render_layout(destination,label,d,state,widths,config):
    ids,x,y,joint,conditional,p=segments(d,state,widths,config)
    if not len(ids):raise ValueError('No retained pieces to draw')
    xmin,xmax=float(x.min()),float(x.max());ymin,ymax=float(y.min()),float(y.max())
    height=int(config.get('render_height',1000))
    if height<800:raise ValueError('Render height must allow space for annotations')
    plot_height=height-505
    sx=sy=plot_height/max(ymax-ymin,1)
    box=(90,115,90+(xmax-xmin)*sx,115+plot_height)
    width=max(1600,int(np.ceil(box[2]+70)))
    px=box[0]+(x-xmin)*sx;py=box[3]-(y-ymin)*sy
    raster=TiledRaster(width,height)
    pdf=VectorPDF(width,height) if config.get('vector_pdf',False) else None
    draw=DrawingSinks(raster,pdf) if pdf else raster
    map_only=config.get('raster_map_only',False)
    map_scale=(height-1)/max(ymax-ymin,1)
    map_width=max(1,int(np.ceil((xmax-xmin)*map_scale))+1)
    map_draw=TiledRaster(map_width,height) if map_only else None
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;fill:#222} .fiber{vector-effect:non-scaling-stroke;stroke-width:.65;stroke-linecap:round}</style>']
    def text(x,y,t,size=16):
        lines.append(f'<text x="{x}" y="{y}" font-size="{size}">{html.escape(t)}</text>')
        draw.text((x,y-size),t,fill=(30,30,30,255),size=size)
    text(90,30,label+' - global U/V assignment',24)
    text(90,56,f'{len(ids):,} retained pieces / {len(state):,}; broken >50% omitted; no subsampling')
    text(90,80,'H blue; V orange. Opacity = joint P(best winding). Equal U/V scale; base-voxel units.')
    text(5,105,'V (base vx)',12)
    for t in np.linspace(0,1,6):
        gx=box[0]+t*(box[2]-box[0]);gy=box[3]-t*(box[3]-box[1])
        lines.append(f'<path d="M{gx} {box[1]}V{box[3]} M{box[0]} {gy}H{box[2]}" stroke="#eee" fill="none"/>')
        if pdf:
            pdf.line((gx,box[1],gx,box[3]),fill=(238,238,238,255))
            pdf.line((box[0],gy,box[2],gy),fill=(238,238,238,255))
        text(gx-20,box[3]+22,f'{xmin+t*(xmax-xmin):.0f}',12)
        text(5,gy+4,f'{ymin+t*(ymax-ymin):.0f}',12)
    lines.append('<g id="retained-fiber-pieces">')
    for n,idx in enumerate(ids):
        rgb=(27,108,168) if d['family'][idx]==0 else (208,105,22)
        color='#%02x%02x%02x'%rgb
        lines.append(f'<line class="fiber" id="piece-{idx}" x1="{px[n,0]:.3f}" y1="{py[n,0]:.3f}" x2="{px[n,1]:.3f}" y2="{py[n,1]:.3f}" stroke="{color}" opacity="{joint[n]:.9f}"/>')
        raster.line((px[n,0],py[n,0],px[n,1],py[n,1]),fill=(*rgb,int(round(255*joint[n]))),width=1)
        if pdf:pdf.line((px[n,0],py[n,0],px[n,1],py[n,1]),fill=(*rgb,255*joint[n]),width=.65)
        if map_draw:
            map_draw.line(((x[n,0]-xmin)*map_scale,(ymax-y[n,0])*map_scale,
                           (x[n,1]-xmin)*map_scale,(ymax-y[n,1])*map_scale),
                          fill=(*rgb,int(round(255*joint[n]))),width=1)
    lines.append('</g>')
    # Same cumulative fitted widths as hard_geometry; these are global bucket
    # boundaries, not per-piece anchor phases or an equal-width approximation.
    boundaries=np.r_[0.,np.cumsum(widths,dtype=np.float64)]
    visible=np.flatnonzero((boundaries>=xmin)&(boundaries<=xmax))
    lines.append('<g id="winding-boundaries">')
    last_label=-np.inf
    for index in visible:
        u_boundary=float(boundaries[index]);bx=box[0]+(u_boundary-xmin)*sx
        lines.append(f'<line class="winding-boundary" data-boundary="{index}" data-u="{u_boundary:.12g}" x1="{bx:.8f}" x2="{bx:.8f}" y1="{box[1]}" y2="{box[3]}" stroke="#555" stroke-opacity=".6" stroke-width=".8" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>')
        for by in range(box[1],box[3],8):
            draw.line((bx,by,bx,min(by+4,box[3])),fill=(85,85,85,153),width=1)
        if map_draw:
            mx=(u_boundary-xmin)*map_scale
            for my in range(0,height,8):map_draw.line((mx,my,mx,min(my+4,height-1)),fill=(85,85,85,153),width=1)
        if bx-last_label>=55:
            text(bx+2,box[1]-5,f'W{index}',10);last_label=bx
    lines.append('</g>')
    text(770,height-340,'U (base voxels)')
    text(90,height-340,'Dashed: fitted winding boundaries (W index)',12)
    bins=np.linspace(0,1,21);cj,_=np.histogram(joint,bins);cc,_=np.histogram(conditional,bins)
    histogram_start=len(lines)
    hist_draw=TiledRaster(1600,350)
    # Share histogram drawing with the vector figure, translated to its own TIFF.
    class HistogramSink:
        def __getattr__(self,name):
            def forward(coords,*args,**kwargs):
                shifted=tuple(v-(height-350 if i%2 else 0) for i,v in enumerate(coords))
                getattr(hist_draw,name)(shifted,*args,**kwargs)
            return forward
    original_draw=draw
    draw=DrawingSinks(draw,HistogramSink())
    for col,(title,counts) in enumerate([('Joint best-bucket probability (also opacity)',cj),('Best winding probability conditional on not broken',cc)]):
        left=90+col*760;top=height-280;bottom=height-70;plotw=670
        text(left,height-305,title,16)
        percentages=100*counts/len(ids);maxpct=max(float(percentages.max()),1)
        for b,value in enumerate(percentages):
            bx=left+b*plotw/20;bh=value/maxpct*(bottom-top)
            lines.append(f'<rect x="{bx:.2f}" y="{bottom-bh:.2f}" width="{plotw/20-2:.2f}" height="{bh:.2f}" fill="#4079ad"/>')
            draw.rectangle((bx,bottom-bh,bx+plotw/20-2,bottom),fill=(64,121,173,255))
        for t in np.linspace(0,1,6):text(left+t*plotw-10,bottom+20,f'{100*t:.0f}%',12)
        text(left,top-5,f'Y: % of retained pieces; top={maxpct:.2f}%',12)
    text(90,height-10,'Confidence is model probability, not calibrated correctness.',14)
    draw=original_draw
    histogram_svg='<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="350" viewBox="0 0 1600 350"><rect width="100%" height="100%" fill="white"/><g transform="translate(0 '+str(350-height)+')">'+ '\n'.join(lines[histogram_start:])+'</g></svg>\n'
    lines.append('</svg>');destination.write_text('\n'.join(lines)+'\n')
    (map_draw if map_draw else raster).save(destination.with_suffix('.tif'))
    if map_only:
        hist_draw.save(destination.with_name('histograms.tif'))
        destination.with_name('histograms.svg').write_text(histogram_svg)
    if pdf:pdf.save(destination.with_suffix('.pdf'))
    # The TIFF replaces the old independently scaled PNG; don't leave a stale
    # preview next to the current layout under the same basename.
    destination.with_suffix('.png').unlink(missing_ok=True)
    return dict(retained_pieces=len(ids),broken_pieces=len(state)-len(ids),bounds=[xmin,xmax,ymin,ymax],
        raster_size=[map_width if map_only else width,height],pixels_per_base_voxel=sx,
        raster_pixels_per_base_voxel=map_scale if map_only else sx,raster_map_only=map_only,
        vector_size=[width,height],raster_compression='LZW',
        joint_quantiles=np.quantile(joint,[0,.05,.25,.5,.75,.95,1]).tolist(),
        conditional_quantiles=np.quantile(conditional,[0,.05,.25,.5,.75,.95,1]).tolist(),
        joint_mean=float(joint.mean()),conditional_mean=float(conditional.mean()),
        conditional_over_90_percent=float(100*np.mean(conditional>=.9)),
        conditional_over_99_percent=float(100*np.mean(conditional>=.99)),
        winding_boundaries=boundaries.tolist(),visible_boundary_indices=visible.tolist(),
        bins=bins.tolist(),joint_counts=cj.tolist(),conditional_counts=cc.tolist())


def solve(config):
    import torch
    from .fiber_uv_categorical import conditional_costs
    from . import fiber_uv_categorical
    root=Path(config['run_root']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'summary.json').read_text());cfg=summary['config']
    with np.load(cfg['tensor_cache']) as src:d={key:src[key] for key in src.files}
    with np.load(root/'solution.npz') as src:
        state=src['state'].copy();widths=src['widths'].copy()
        for key in ('source_id','start','end','family'):
            if not np.array_equal(d[key],src[key]):raise ValueError('Piece identity mismatch')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    t=torch.tensor(state,device='cuda');w=torch.tensor(widths,device='cuda',dtype=torch.float64)
    keys=('family','phase','z','degree','node_ds','node_phase','node_z','i','j','ds_a','ds_b',
          'phase_a','phase_b','phase_delta','distance','sign','links')
    data={key:torch.as_tensor(d[key],device='cuda',dtype=torch.long if key in ('family','i','j') else torch.float64) for key in keys}
    table=torch.tensor(uv.make_distance_table(float(d['distance'].max()),cfg.get('distance_sigma',.5)),device='cuda')
    with torch.no_grad():
        _,energy=conditional_costs(t,w,data,table,cfg,True)
        initial=torch.tensor(d['width_initial'],device='cuda',dtype=torch.float64)
        constant=uv.offset_regularization(t,100,cfg)+cfg.get('width_prior',.1)*((w/initial-1).square().mean())
        constant+=cfg.get('width_smooth',.01)*torch.diff(w/initial).square().mean()
        p=t[:,:101].softmax(1)
        negative_entropy=(p*p.clamp_min(1e-30).log()).sum(1).mean()*cfg.get('target_temperature',.25)
        energy=float(energy);constant=float(constant);negative_entropy=float(negative_entropy)
    rows=progression(json.loads((root/'history.json').read_text()),energy,constant)
    vis=render_layout(out/'layout.svg',config['label'],d,state,widths,cfg)
    result=dict(run_root=str(root),label=config['label'],solution_sha256=uv.sha256_file(root/'solution.npz'),
        tensor_cache_sha256=uv.sha256_file(cfg['tensor_cache']),sweeps=rows,confidence=vis,
        final_negative_entropy_term=negative_entropy,final_free_energy=energy+constant+negative_entropy,
        expected_cost_definition='pair expectation + rejection + geometric priors + continuation + frozen geometry constants; no entropy',
        source_sha256=uv.sha256_file(__file__),geometry_sha256=uv.sha256_file(uv.__file__),
        cost_module_sha256=uv.sha256_file(fiber_uv_categorical.__file__))
    uv.atomic_write(out/'summary.json',result,durable=False)
    print(json.dumps(dict(label=config['label'],layout=str(out/'layout.svg'),confidence=vis,final_energy=energy+constant)),flush=True)


def render_progress(destination,reports):
    width,height=1400,820;lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:sans-serif}}</style>']
    for panel,key in enumerate(('violations','global_expected_cost')):
        left=100+panel*690;top=100;bottom=690;pw=570
        maxv=max(row[key] for r in reports for row in r['sweeps']);maxs=max(row['sweep'] for r in reports for row in r['sweeps'])
        lines.append(f'<text x="{left}" y="45" font-size="22">{key.replace("_"," ")}</text>')
        for t in np.linspace(0,1,6):
            yy=bottom-t*(bottom-top)
            lines.append(f'<path d="M{left} {yy}h{pw}" stroke="#ddd"/><text x="{left-80}" y="{yy}">{maxv*t:.2f}</text>')
        for n,r in enumerate(reports):
            color=('#2166ac','#d6604d')[n%2]
            points=' '.join(f'{left+pw*row["sweep"]/max(maxs,1):.2f},{bottom-(bottom-top)*row[key]/max(maxv,1e-20):.2f}' for row in r['sweeps'])
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{left}" y="{750+n*25}" fill="{color}">{html.escape(r["label"])}</text>')
        for step in range(maxs+1):lines.append(f'<text x="{left+pw*step/max(maxs,1)}" y="715">{step}</text>')
        lines.append(f'<text x="{left+pw/2}" y="740">Completed sweeps</text>')
    lines.append('</svg>');destination.write_text('\n'.join(lines)+'\n')


def run(config):
    out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True);reports=[]
    if config.get('layout_only',False):
        for item in config['solutions']:
            root=Path(item['run_root']);destination=out/item['label'].lower();destination.mkdir(parents=True,exist_ok=True)
            cfg=json.loads((root/'summary.json').read_text())['config']
            with np.load(cfg['tensor_cache']) as src:d={key:src[key] for key in ('phase','z','family','node_ds','source_id','start','end')}
            with np.load(root/'solution.npz') as src:
                for key in ('source_id','start','end','family'):
                    if not np.array_equal(d[key],src[key]):raise ValueError('Piece identity mismatch')
                vis=render_layout(destination/'layout.svg',item['label'],d,src['state'],src['widths'],
                    dict(cfg,**{key:config[key] for key in ('render_height','raster_map_only','vector_pdf') if key in config}))
            result=dict(run_root=str(root),confidence=vis,solution_sha256=uv.sha256_file(root/'solution.npz'),
                tensor_cache_sha256=uv.sha256_file(cfg['tensor_cache']),render_source_sha256=uv.sha256_file(__file__))
            uv.atomic_write(destination/'summary.json',result,durable=False);reports.append(result)
            print(json.dumps(dict(label=item['label'],confidence=vis)),flush=True)
        uv.atomic_write(out/'results.json',reports,durable=False)
        return reports
    for item in config['solutions']:
        task=dict(item,output_root=str(out/item['label'].lower()),gpu_manager=config['gpu_manager'])
        if config.get('render_only',False):
            destination=Path(task['output_root']);root=Path(item['run_root'])
            result=json.loads((destination/'summary.json').read_text())
            cfg=json.loads((root/'summary.json').read_text())['config']
            if uv.sha256_file(root/'solution.npz')!=result['solution_sha256'] or uv.sha256_file(cfg['tensor_cache'])!=result['tensor_cache_sha256']:
                raise ValueError('Visualization source changed')
            with np.load(cfg['tensor_cache']) as src:d={key:src[key] for key in src.files}
            with np.load(root/'solution.npz') as src:
                result['confidence']=render_layout(destination/'layout.svg',item['label'],d,src['state'],src['widths'],cfg)
            result['render_source_sha256']=uv.sha256_file(__file__)
            uv.atomic_write(destination/'summary.json',result,durable=False)
            reports.append(result)
        else:reports.append(uv.launch_tensor_solve(task,__name__,Path(__file__)))
    render_progress(out/'progression.svg',reports)
    with (out/'progression.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['solution',*reports[0]['sweeps'][0]])
        writer.writeheader()
        for r in reports:
            for row in r['sweeps']:writer.writerow(dict(solution=r['label'],**row))
    lines=['# Full U/V sweep progression','',
        'Global expected cost includes frozen geometry constants, excludes entropy. Lower is better.',
        'Costs belong to the state at the listed sweep; pre-update history was shifted back one sweep.',
        'No optimization rerun. Final cost evaluated from saved states using existing GPU cost code.','',
        '| Sweep | CCW violations | CCW global cost | CW violations | CW global cost |',
        '|---|---:|---:|---:|---:|']
    for a,b in zip(reports[0]['sweeps'],reports[1]['sweeps']):
        lines.append(f'|{a["sweep"]}|{a["violations"]:,}|{a["global_expected_cost"]:.6f}|{b["violations"]:,}|{b["global_expected_cost"]:.6f}|')
    for r in reports:
        c=r['confidence'];lines+=['',f'## {r["label"]}',
            f'Retained pieces:{c["retained_pieces"]}; rejected omitted:{c["broken_pieces"]}.',
            f'Conditional confidence median:{c["conditional_quantiles"][3]:.6f}; >=90%:{c["conditional_over_90_percent"]:.4f}%; >=99%:{c["conditional_over_99_percent"]:.4f}%.',
            f'Joint confidence median:{c["joint_quantiles"][3]:.6f}.',
            f'Final negative entropy term:{r["final_negative_entropy_term"]:.6f}; variational free energy:{r["final_free_energy"]:.6f}.']
    (out/'progression.md').write_text('\n'.join(lines)+'\n');uv.atomic_write(out/'results.json',reports,durable=False)
    return dict(output_root=str(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--unmanaged',action='store_true');p.add_argument('--devices')
    a=p.parse_args()
    if not a.unmanaged:raise RuntimeError('Use approved managed runner')
    solve(json.loads(a.config.read_text()))
