"""Read-only saved U/V solution plots and correctly aligned global cost curves."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from . import fiber_uv_layout as uv


def segments(d,state,widths,config):
    if config.get('saved_model') == 'implicit_discrete_winding_v1':
        import torch
        from .fiber_implicit_winding import hypotheses
        s=torch.as_tensor(state,dtype=torch.float64)
        w=torch.as_tensor(widths,dtype=torch.float64)
        sigma=float(config['winding_sigma']);bound=config.get('sigma_support_max')
        if config.get('learned_winding_sigma'):
            sigma=torch.as_tensor(config['_saved_winding_sigma'],dtype=s.dtype)
            bound=config.get('sigma_max',1.)
        _,logp=hypotheses(s,sigma,bound)
        enabled=s[:,3].sigmoid().numpy()
        conditional=logp.exp().amax(1).numpy()
        ids=np.flatnonzero(enabled>=.5)
        q=torch.floor(s[:,0]+.5)+s[:,1]
        boundaries=torch.cat((w.new_zeros(1),w.cumsum(0)))
        anchor=uv.q_to_u(q,w,boundaries).numpy()
        ds=d['node_ds'][ids][:,[0,2]]
        x=anchor[ids,None]+(d['family'][ids]==0)[:,None]*ds
        y=state[ids,2,None]+(d['family'][ids]==1)[:,None]*ds
        return ids,x,y,enabled[ids]*conditional[ids],conditional[ids],None
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
        font=kwargs.get('font',ImageFont.load_default())
        b=font.getbbox(value);x,y=coords
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


def dashed_segments(a,b,dash=6.,gap=4.):
    """Screen-space dashes shared by TIFF and PDF output."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    length=float(np.linalg.norm(b-a))
    if length==0:return
    direction=(b-a)/length
    for start in np.arange(0,length,dash+gap):
        yield (* (a+direction*start),* (a+direction*min(start+dash,length)))


def render_layout(destination,label,d,state,widths,config):
    ids,x,y,joint,conditional,p=config.get('_geometry') or segments(d,state,widths,config)
    if not len(ids):raise ValueError('No retained pieces to draw')
    xmin,xmax,ymin,ymax=config.get('_bounds',(min(map(np.min,x)),max(map(np.max,x)),min(map(np.min,y)),max(map(np.max,y))))
    xmin,xmax,ymin,ymax=map(float,(xmin,xmax,ymin,ymax))
    height=int(config.get('render_height',1000))
    if height<800:raise ValueError('Render height must allow space for annotations')
    plot_height=height-505
    sx=sy=plot_height/max(ymax-ymin,1)
    box=(90,115,90+(xmax-xmin)*sx,115+plot_height)
    width=max(1600,int(np.ceil(box[2]+70)))
    px=[box[0]+(xx-xmin)*sx for xx in x];py=[box[3]-(yy-ymin)*sy for yy in y]
    raster=TiledRaster(width,height)
    pdf=VectorPDF(width,height) if config.get('vector_pdf',False) else None
    draw=DrawingSinks(raster,pdf) if pdf else raster
    map_only=config.get('raster_map_only',False)
    map_scale=(height-1)/max(ymax-ymin,1)
    map_width=max(1,int(np.ceil((xmax-xmin)*map_scale))+1)
    map_draw=TiledRaster(map_width,height) if map_only else None
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;fill:#222} .fiber{vector-effect:non-scaling-stroke;stroke-width:'+str(float(config.get('fiber_vector_line_width',.65)))+';stroke-linecap:round}</style>']
    def text(x,y,t,size=16):
        lines.append(f'<text x="{x}" y="{y}" font-size="{size}">{html.escape(t)}</text>')
        draw.text((x,y-size),t,fill=(30,30,30,255),size=size)
    text(90,30,label+' - global U/V assignment',24)
    text(90,56,config.get('_cohort_label',f'{len(ids):,} retained pieces / {len(state):,}; broken >50% omitted; no subsampling'))
    text(90,80,config.get('_legend','H blue; V orange. Opacity = joint P(best winding). Equal U/V scale; base-voxel units.'))
    text(5,105,'V (base vx)',12)
    for t in np.linspace(0,1,6):
        gx=box[0]+t*(box[2]-box[0]);gy=box[3]-t*(box[3]-box[1])
        lines.append(f'<path d="M{gx} {box[1]}V{box[3]} M{box[0]} {gy}H{box[2]}" stroke="#eee" fill="none"/>')
        if pdf:
            pdf.line((gx,box[1],gx,box[3]),fill=(238,238,238,255))
            pdf.line((box[0],gy,box[2],gy),fill=(238,238,238,255))
        text(gx-20,box[3]+22,f'{xmin+t*(xmax-xmin):.0f}',12)
        text(5,gy+4,f'{ymin+t*(ymax-ymin):.0f}',12)
    # Context fibers never enter foreground statistics or annotation cohorts.
    background=config.get('_background_geometry')
    if background is not None:
        lines.append('<g id="individual-probability-background">')
        for xx,yy in zip(background[1],background[2]):
            coords=' '.join(f'{box[0]+(a-xmin)*sx:.3f},{box[3]-(b-ymin)*sy:.3f}' for a,b in zip(xx,yy))
            lines.append(f'<polyline points="{coords}" fill="none" stroke="#cbdfee" stroke-width="0.3"/>')
            for k in range(len(xx)-1):
                line=(box[0]+(xx[k]-xmin)*sx,box[3]-(yy[k]-ymin)*sy,
                      box[0]+(xx[k+1]-xmin)*sx,box[3]-(yy[k+1]-ymin)*sy)
                raster.line(line,fill=(203,223,238,255),width=1)
                if pdf:pdf.line(line,fill=(203,223,238,255),width=.3)
                if map_draw:
                    map_draw.line(((xx[k]-xmin)*map_scale,(ymax-yy[k])*map_scale,
                        (xx[k+1]-xmin)*map_scale,(ymax-yy[k+1])*map_scale),fill=(203,223,238,255),width=1)
        lines.append('</g>')
    lines.append('<g id="context-connections">')
    for a,b in config.get('_context_connections',[]):
        aa=(box[0]+(a[0]-xmin)*sx,box[3]-(a[1]-ymin)*sy)
        bb=(box[0]+(b[0]-xmin)*sx,box[3]-(b[1]-ymin)*sy)
        lines.append(f'<line x1="{aa[0]}" y1="{aa[1]}" x2="{bb[0]}" y2="{bb[1]}" stroke="#8fb4d0" stroke-width="1.5" stroke-dasharray="6 4"/>')
        for segment in dashed_segments(aa,bb):
            raster.line(segment,fill=(143,180,208,255),width=2)
            if pdf:pdf.line(segment,fill=(143,180,208,255),width=1.5)
        if map_draw:
            aa=((a[0]-xmin)*map_scale,(ymax-a[1])*map_scale)
            bb=((b[0]-xmin)*map_scale,(ymax-b[1])*map_scale)
            for segment in dashed_segments(aa,bb):map_draw.line(segment,fill=(143,180,208,255),width=2)
    lines.append('</g>')
    # Actual constraint links are a distinct dark background layer, not leaders.
    lines.append('<g id="constraint-connections">')
    for a,b in config.get('_connections',[]):
        ax,ay=box[0]+(a[0]-xmin)*sx,box[3]-(a[1]-ymin)*sy
        bx,by=box[0]+(b[0]-xmin)*sx,box[3]-(b[1]-ymin)*sy
        lines.append(f'<line class="constraint" x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="#000" stroke-width="2"/>')
        draw.line((ax,ay,bx,by),fill=(0,0,0,255),width=2)
        if map_draw:
            map_draw.line(((a[0]-xmin)*map_scale,(ymax-a[1])*map_scale,
                           (b[0]-xmin)*map_scale,(ymax-b[1])*map_scale),fill=(0,0,0,255),width=3)
    lines.append('</g>')
    lines.append('<g id="sign-violation-underlays">')
    for n in config.get('_red_underlay_indices',[]):
        coords=' '.join(f'{xx:.3f},{yy:.3f}' for xx,yy in zip(px[n],py[n]))
        vw=2*float(config.get('fiber_vector_line_width',.65))
        lines.append(f'<polyline id="sign-underlay-{ids[n]}" points="{coords}" fill="none" stroke="#f9d2d2" stroke-width="{vw}" stroke-linecap="round" vector-effect="non-scaling-stroke"/>')
        for k in range(len(x[n])-1):
            line=(px[n][k],py[n][k],px[n][k+1],py[n][k+1])
            raster.line(line,fill=(249,210,210,255),width=2)
            if pdf:pdf.line(line,fill=(249,210,210,255),width=vw)
            if map_draw:
                map_draw.line(((x[n][k]-xmin)*map_scale,(ymax-y[n][k])*map_scale,
                               (x[n][k+1]-xmin)*map_scale,(ymax-y[n][k+1])*map_scale),
                              fill=(249,210,210,255),width=2*int(config.get('fiber_line_width',1)))
    lines.append('</g>')
    lines.append('<g id="retained-fiber-pieces">')
    for n in config.get('_draw_order',range(len(ids))):
        idx=ids[n]
        rgb=config['_colors'][n] if '_colors' in config else ((27,108,168) if d['family'][idx]==0 else (208,105,22))
        color='#%02x%02x%02x'%rgb
        dotted=bool(config.get('_dotted', [False]*len(ids))[n])
        dash_style=' stroke-dasharray="1 4"' if dotted else ''
        coords=' '.join(f'{xx:.3f},{yy:.3f}' for xx,yy in zip(px[n],py[n]))
        if len(x[n])==2:
            lines.append(f'<line class="fiber" id="piece-{idx}" x1="{px[n][0]:.3f}" y1="{py[n][0]:.3f}" x2="{px[n][1]:.3f}" y2="{py[n][1]:.3f}" stroke="{color}" opacity="{joint[n]:.9f}"{dash_style}/>')
        else:
            lines.append(f'<polyline class="fiber" id="piece-{idx}" points="{coords}" fill="none" stroke="{color}" opacity="{joint[n]:.9f}"{dash_style}/>')
        for k in range(len(x[n])-1):
            line=(px[n][k],py[n][k],px[n][k+1],py[n][k+1])
            for part in (dashed_segments(line[:2],line[2:],dash=1,gap=4) if dotted else [line]):
                raster.line(part,fill=(*rgb,int(round(255*joint[n]))),width=1)
                if pdf:pdf.line(part,fill=(*rgb,255*joint[n]),width=float(config.get('fiber_vector_line_width',.65)))
            if map_draw:
                line=((x[n][k]-xmin)*map_scale,(ymax-y[n][k])*map_scale,
                      (x[n][k+1]-xmin)*map_scale,(ymax-y[n][k+1])*map_scale)
                for part in (dashed_segments(line[:2],line[2:],dash=1,gap=4) if dotted else [line]):
                    map_draw.line(part,fill=(*rgb,int(round(255*joint[n]))),width=int(config.get('fiber_line_width',1)))
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
    # Optional diagnostic labels use physical UV coordinates in all output sinks.
    annotation_items=config.get('_annotations',[])
    if config.get('diagnostic_label_layout'):
        from PIL import ImageFont
        from .fiber_label_layout import place_labels
        # Previously vector labels were 9pt and TIFF labels used an ~11px bitmap font.
        label_size=float(config.get('diagnostic_label_font_size',6))
        font=ImageFont.truetype('DejaVuSans.ttf',max(1,round(label_size)))
        def measure(value):
            b=font.getbbox(value);return b[2]-b[0],b[3]-b[1]
        vector=place_labels(annotation_items,lambda p:(box[0]+(p[0]-xmin)*sx,box[3]-(p[1]-ymin)*sy),
                            box,lambda value:(len(value)*label_size*.5,label_size))
        for entry in vector:
            ax,ay=entry['anchor'];lx,ly,rx,by=entry['box']
            tx=min(max(ax,lx),rx);ty=min(max(ay,ly),by)
            lines.append(f'<line x1="{ax}" y1="{ay}" x2="{tx}" y2="{ty}" stroke="#aaa" stroke-width=".3"/>')
            draw.line((ax,ay,tx,ty),fill=(160,160,160,120),width=1)
            text(lx+1,ly+label_size*.92,entry['item']['text'],label_size*.75)
        if map_draw:
            raster_labels=place_labels(annotation_items,lambda p:((p[0]-xmin)*map_scale,(ymax-p[1])*map_scale),
                                       (0,0,map_width-1,height-1),measure)
            for entry in raster_labels:
                ax,ay=entry['anchor'];lx,ly,rx,by=entry['box'];value=entry['item']['text']
                map_draw.line((ax,ay,min(max(ax,lx),rx),min(max(ay,ly),by)),fill=(160,160,160,120),width=1)
                b=font.getbbox(value)
                map_draw.text((lx+1-b[0],ly+1-b[1]),value,font=font,fill=(30,30,30,255))
        annotation_items=[]
    for item in annotation_items:
        xx,yy=item['uv'];tx=box[0]+(xx-xmin)*sx;ty=box[3]-(yy-ymin)*sy
        text(tx+3,ty-3,item['text'],9)
        if map_draw:
            map_draw.text(((xx-xmin)*map_scale+3,(ymax-yy)*map_scale-12),
                          item['text'],fill=(30,30,30,255))
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
        constant=uv.offset_regularization(t,100,cfg)
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


def deformation_geometry(d,saved,config):
    """Original/deformed polylines and whole-source maximum offset slopes."""
    cfg=dict(config,saved_model='implicit_discrete_winding_v1')
    ids,x,y,joint,conditional,_=segments(d,saved['state'],saved['widths'],cfg)
    keep=saved['retained'][ids].astype(bool)
    ids,x,y,joint,conditional=ids[keep],x[keep],y[keep],joint[keep],conditional[keep]
    source=saved['v_offset_source'];arc=saved['v_offset_arc']
    l,r=saved['v_offset_left'],saved['v_offset_right'];gap=arc[r]-arc[l]
    urate=np.abs(saved['u_offsets'][r]-saved['u_offsets'][l])/gap
    vrate=np.abs(saved['v_offsets'][r]-saved['v_offsets'][l])/gap
    rates={}
    for sid in np.unique(d['source_id'][ids]):
        mask=source[l]==sid
        rates[int(sid)]=(float(np.max(urate[mask],initial=0)),float(np.max(vrate[mask],initial=0)))
    original_x=[];original_y=[];deformed_x=[];deformed_y=[]
    for n,idx in enumerate(ids):
        sid=d['source_id'][idx];lo,hi=d['start'][idx],d['end'][idx]
        nodes=np.flatnonzero(source==sid)
        knots=np.unique(np.r_[lo,arc[nodes][(arc[nodes]>lo)&(arc[nodes]<hi)],hi])
        t=(knots-lo)/(hi-lo)
        xx=x[n,0]+t*(x[n,1]-x[n,0]);yy=y[n,0]+t*(y[n,1]-y[n,0])
        du=np.interp(knots,arc[nodes],saved['u_offsets'][nodes]) if len(nodes) else np.zeros(len(knots))
        dv=np.interp(knots,arc[nodes],saved['v_offsets'][nodes]) if len(nodes) else np.zeros(len(knots))
        original_x.append(xx);original_y.append(yy);deformed_x.append(xx+du);deformed_y.append(yy+dv)
    return ((ids,original_x,original_y,joint,conditional,None),
        (ids,deformed_x,deformed_y,joint,conditional,None),rates)


def render_deformation_comparison(config):
    root=Path(config['deformation_comparison']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'summary.json').read_text());cfg=summary['config']
    with np.load(cfg['tensor_cache']) as f:d={k:f[k] for k in f.files}
    with np.load(root/'solution.npz') as f:saved={k:f[k] for k in f.files}
    for key in ('source_id','start','end','family'):
        if not np.array_equal(saved[key],d[key]):raise ValueError('Visualization identity mismatch')
    before,after,rates=deformation_geometry(d,saved,cfg)
    bounds=[min(np.min(x) for x in before[1]+after[1]),max(np.max(x) for x in before[1]+after[1]),
        min(np.min(y) for y in before[2]+after[2]),max(np.max(y) for y in before[2]+after[2])]
    ids=before[0];maximum=np.array([max(rates[int(d['source_id'][i])]) for i in ids])
    category=(maximum>.1).astype(int)+(maximum>.5).astype(int)
    base=[(90,125,160) if d['family'][i]==0 else (155,165,175) for i in ids]
    colors=[(220,35,45) if c==2 else (240,145,20) if c==1 else base[n] for n,c in enumerate(category)]
    common=dict(cfg,render_height=config.get('render_height',1000),raster_map_only=True,vector_pdf=True,
        fiber_line_width=2,_bounds=bounds,_cohort_label=f'{len(ids)} fixed confident pieces; identical cohort and scale in both views')
    reports={}
    for name,geometry,palette in [('original',before,base),('deformed',after,colors)]:
        legend=('Orange: max U/V offset slope >10%; red: >50%. Whole-source highlights; base voxels.' if name=='deformed'
                else 'Original regular solve; H blue, V gray. Fixed >50% joint-confidence cohort.')
        reports[name]=render_layout(out/(name+'.svg'),name,d,saved['state'],saved['widths'],
            dict(common,_geometry=geometry,_colors=palette,_legend=legend,_draw_order=np.argsort(category,kind='stable')))
    with (out/'fiber_deformation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['source_id','max_u_rate','max_v_rate','max_component_rate','highlight','piece_ids'])
        for sid,(u,v) in sorted(rates.items(),key=lambda pair:max(pair[1]),reverse=True):
            writer.writerow([sid,u,v,max(u,v),'red' if max(u,v)>.5 else 'orange' if max(u,v)>.1 else 'normal',
                ' '.join(map(str,ids[d['source_id'][ids]==sid]))])
    result=dict(views=reports,whole_source_count=len(rates),
        orange_sources=sum(.1<max(r)<=.5 for r in rates.values()),red_sources=sum(max(r)>.5 for r in rates.values()),
        orange_pieces=int(np.sum(category==1)),red_pieces=int(np.sum(category==2)),
        solution_sha256=uv.sha256_file(root/'solution.npz'),render_source_sha256=uv.sha256_file(__file__))
    uv.atomic_write(out/'summary.json',result,durable=False)
    return result


def run(config):
    if config.get('bucket_comparison'):
        from .fiber_bucket_uv_visualization import run as render_bucket
        return render_bucket(config)
    if config.get('deformation_comparison'):return render_deformation_comparison(config)
    out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True);reports=[]
    if config.get('layout_only',False):
        for item in config['solutions']:
            root=Path(item['run_root']);destination=out/item['label'].lower();destination.mkdir(parents=True,exist_ok=True)
            saved=json.loads((root/'summary.json').read_text())
            cfg=dict(saved['config'],saved_model=saved.get('model'))
            with np.load(cfg['tensor_cache']) as src:d={key:src[key] for key in ('phase','z','family','node_ds','source_id','start','end')}
            with np.load(root/'solution.npz') as src:
                for key in ('source_id','start','end','family'):
                    if not np.array_equal(d[key],src[key]):raise ValueError('Piece identity mismatch')
                if cfg.get('learned_winding_sigma'):cfg['_saved_winding_sigma']=src['winding_sigma']
                vis=render_layout(destination/'layout.svg',item['label'],d,src['state'],src['widths'],
                    dict(cfg,**{key:config[key] for key in ('render_height','raster_map_only','vector_pdf','fiber_line_width') if key in config}))
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
