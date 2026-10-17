"""Render numeric orthogonal slices and a self-contained benchmark index."""
import html
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bench_augment_native import arrays
from augment_catalog import UTILITY_TRANSFORMS


PREVIEW_COLUMNS = 3
PREVIEW_TILE = 360
PREVIEW_GAP = 24
PREVIEW_TOP = 210
PREVIEW_ROW_HEIGHT = PREVIEW_TILE + 62
PREVIEW_FOOTER = 65


def export_axis_previews(source, directory, name):
    """Export paired original/augmented Z, Y, X panels from a numeric preview.

    Keep its selected planes, display windows and localized-effect annotations
    intact. Reject other layouts rather than silently exporting the wrong area.
    """
    expected=(PREVIEW_COLUMNS*PREVIEW_TILE+(PREVIEW_COLUMNS+1)*PREVIEW_GAP,
              PREVIEW_TOP+2*PREVIEW_ROW_HEIGHT+PREVIEW_FOOTER)
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    with Image.open(source) as canvas:
        if canvas.size!=expected:raise ValueError(f'expected a three-axis preview: {source}')
        for axis,label in enumerate(('z','y','x')):
            left=PREVIEW_GAP+axis*(PREVIEW_TILE+PREVIEW_GAP)
            path=directory/f'{name}-{label}.webp'
            canvas.crop((left,PREVIEW_TOP,left+PREVIEW_TILE,PREVIEW_TOP+2*PREVIEW_ROW_HEIGHT)).save(path,lossless=True,method=6)
            paths.append(path)
    return paths


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def preview_volumes(case, source, result):
    output=dict(arrays(result))[case.output]
    key='mask' if case.sample=='labels' else ('surface' if case.sample=='surface' else 'image')
    original=source[key]
    if case.name=='ToPseudo2D':output=output.reshape(original.shape)
    return key, original, output, original[min(case.channel,original.shape[0]-1)], output[min(case.channel,output.shape[0]-1)]


def effect_strength(case, source, result):
    _,_,_,a,b=preview_volumes(case,source,result)
    if a.shape!=b.shape:return float('inf')
    return int(np.count_nonzero(a!=b))


def effect_slices(original, augmented, *, derived=False, interior_fraction=0):
    """Maximize changed area per axis; ties favor the center, never an empty slice.

    Changed grids are compared at corresponding relative positions, not claimed
    to be registered. Derived outputs select planes with the most output signal.
    An interior search excludes boundary padding from both scoring and selection.
    """
    mapped=original
    if original.shape!=augmented.shape:
        for axis,n in enumerate(augmented.shape):
            mapped=np.take(mapped,np.rint(np.linspace(0,mapped.shape[axis]-1,n)).astype(int),axis=axis)
    difference=np.abs(augmented.astype(np.float32)-mapped.astype(np.float32))
    signal=np.abs(augmented.astype(np.float32)) if derived else difference
    if not 0<=interior_fraction<.5:raise ValueError('interior_fraction must be in [0, .5)')
    margins=[min((n-1)//2,int(n*interior_fraction)) for n in signal.shape]
    signal=signal[tuple(slice(m,n-m) for n,m in zip(signal.shape,margins))]
    indices=[]
    for axis in range(3):
        other=tuple(a for a in range(3) if a!=axis)
        scores=np.count_nonzero(signal,axis=other)
        strongest=scores.max()
        candidates=np.flatnonzero(scores==strongest)
        if strongest:
            magnitudes=signal.sum(axis=other)
            candidates=candidates[magnitudes[candidates]==magnitudes[candidates].max()]
        indices.append(int(candidates[np.argmin(np.abs(candidates-(signal.shape[axis]-1)/2))])+margins[axis])
    original_indices=[round(i*(original.shape[a]-1)/max(1,augmented.shape[a]-1)) for a,i in enumerate(indices)]
    return original_indices,indices,difference


def changed_bounds(delta):
    """Mark small changed regions without obscuring broadly changed images."""
    rows,cols=np.nonzero(delta)
    if not len(rows):return None
    y0,y1=int(rows.min()),int(rows.max())+1
    x0,x1=int(cols.min()),int(cols.max())+1
    if (y1-y0)*(x1-x0)>delta.size/4:return None
    return max(0,x0-3),max(0,y0-3),min(delta.shape[1]-1,x1+3),min(delta.shape[0]-1,y1+3)


def preview(path, case, source, result, size, parameters, seed):
    key,original,output,a,b=preview_volumes(case,source,result)
    derived=case.name in ('DistanceTransform','StructureTensor','SurfacePerturb')
    panels=[]
    selection='2D input'; metadata={'seed':seed}; highlights={}
    if a.ndim==3 and b.ndim==3:
        interior_fraction=.25 if case.name=='Elastic' else 0
        before,after,delta=effect_slices(a,b,derived=derived,interior_fraction=interior_fraction)
        selection='most affected slices' if np.any(delta) else 'no voxel-value change for this sample'
        if interior_fraction and np.any(delta):selection='most affected interior slices (boundary padding excluded from selection)'
        if derived:selection='slices with strongest derived output'
        if a.shape!=b.shape:selection+=' · coordinates mapped by relative grid position'
        panels=[(f'Original {key} · {axis}={index}',np.take(a,index,axis=j),False)
                for j,(axis,index) in enumerate(zip(('Z','Y','X'),before))]
        panels += [(f'Augmented {case.output} · {axis}={index}',np.take(b,index,axis=j),True)
                   for j,(axis,index) in enumerate(zip(('Z','Y','X'),after))]
        if a.shape==b.shape and not derived:
            for axis,index in enumerate(after):
                box=changed_bounds(np.take(delta,index,axis=axis))
                if box is not None:highlights[axis]=box
        metadata=dict(original_indices=before,augmented_indices=after,
                      changed_voxels=[int(np.count_nonzero(np.take(delta,index,axis=j))) for j,index in enumerate(after)],
                      selection=selection,seed=seed)
        if interior_fraction:metadata['interior_fraction']=interior_fraction
    else:
        panels=[('Original · central Z plane / 2D input',a,False),('Augmented · 2D output',b,True)]
        if case.name=='SurfacePerturb':
            delta=np.sqrt(np.sum((output-original)**2,axis=0))
            panels.append(('Coordinate displacement magnitude',delta,True))
    joined=[p for _,p,_ in panels]
    lo=min(float(v.min()) for v in joined);hi=max(float(v.max()) for v in joined)
    if not derived and case.sample in ('image','both','signed','channels','strip') and case.name not in ('Normalize',):
        lo=min(0.,lo);hi=max(1.,hi)
    widths=PREVIEW_COLUMNS;tile=PREVIEW_TILE;gap=PREVIEW_GAP;top=PREVIEW_TOP;rows=(len(panels)+widths-1)//widths
    canvas=Image.new('RGB',(widths*tile+(widths+1)*gap,top+rows*PREVIEW_ROW_HEIGHT+PREVIEW_FOOTER),'#101827')
    draw=ImageDraw.Draw(canvas)
    draw.text((gap,20),f'{case.name}  |  {size}³ input'+(' · 2D companion' if a.ndim==2 else ''),font=font(29),fill='white')
    y=65
    for line in textwrap.wrap(parameters,110)[:4]:
        draw.text((gap,y),line,font=font(16),fill='#c6d3e5');y+=23
    draw.text((gap,top-30),f'CPU output · seed {seed} · channel {case.channel} · {selection}',font=font(15),fill='#a4b5ce')
    for i,(title,plane,is_output) in enumerate(panels):
        x=gap+(i%widths)*(tile+gap);y=top+(i//widths)*PREVIEW_ROW_HEIGHT
        low,high=lo,hi
        if derived:
            group=[v for _,v,out in panels if out==is_output]
            low=min(float(v.min()) for v in group);high=max(float(v.max()) for v in group)
            if case.name=='SurfacePerturb' and i==2:low,high=0.,float(plane.max())
        scaled=np.clip((plane-low)/(high-low if high>low else 1),0,1)
        picture=Image.fromarray(np.rint(scaled*255).astype(np.uint8)).convert('RGB')
        factor=min(tile/picture.width,tile/picture.height)
        picture=picture.resize((round(picture.width*factor),round(picture.height*factor)),Image.Resampling.NEAREST)
        draw.text((x,y),title,font=font(16),fill='white')
        left=x+(tile-picture.width)//2;upper=y+29+(tile-picture.height)//2
        canvas.paste(picture,(left,upper))
        if i%3 in highlights:
            x0,y0,x1,y1=highlights[i%3]
            sx=picture.width/plane.shape[1];sy=picture.height/plane.shape[0]
            draw.rectangle((left+x0*sx,upper+y0*sy,left+x1*sx,upper+y1*sy),outline='#ffce66',width=2)
        draw.text((x,y+tile+33),f'window [{low:.4g}, {high:.4g}] · {plane.shape[0]} × {plane.shape[1]}',font=font(14),fill='#a4b5ce')
    footer=case.note or ('Derived output uses different units; windows are labeled. No per-slice autoscaling.' if derived else 'Shared display window; no per-slice contrast adjustment. Z→YX, Y→ZX, X→ZY.')
    if highlights:footer='Gold boxes locate changed regions (annotations). '+footer
    for i,line in enumerate(textwrap.wrap(footer,125)[:2]):
        draw.text((gap,canvas.height-54+i*22),line,font=font(15),fill='#a4b5ce')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return metadata


def gallery(directory, reports, *, include_utilities=False):
    # Keep historical JSON intact; the visible table, cards and CSV share scope.
    reports=[dict(report,results=[row for row in report['results']
             if include_utilities or row['name'] not in UTILITY_TRANSFORMS]) for report in reports]
    rows=[]
    for report in reports:
        for row in sorted(report['results'],key=lambda r:r['name']):
            cells=[]
            for mode in ('cpu','cuda_host','cuda_resident'):
                stats=row.get('timings',{}).get(mode,{})
                cells.append(f"{stats['p50_ms']:.3f}" if stats else '—')
            picture=row.get('preview','')
            label=html.escape(row['name']);params=html.escape(row['parameters']);note=html.escape(row.get('note',''))
            status=html.escape(row.get('error',''))
            rows.append(f'''<article id="{report["size"]}-{label}" data-name="{label.lower()}"><h2>{label} <small>{report['size']}³{' · 2D companion' if row['sample'] in ('strip','surface') else ''}</small></h2>
<p class="params">{params}</p><p>Median ms — CPU <b>{cells[0]}</b> · CUDA with transfers <b>{cells[1]}</b> · CUDA resident <b>{cells[2]}</b></p>
<p>{note} {status}</p><a href="{picture}"><img loading="lazy" src="{picture}" alt="{label}: original and augmented orthogonal slices"></a></article>''')
    by_name={}
    for report in reports:
        for row in report['results']:by_name.setdefault(row['name'],{})[report['size']]=row
    sizes=sorted(r['size'] for r in reports)
    table='<div style="overflow:auto"><table><thead><tr><th>Augmentation</th>'+''.join(f'<th colspan="5">{size}³ · median ms</th>' for size in sizes)+'</tr><tr><th></th>'+''.join('<th>CPU</th><th>CUDA + transfers</th><th>CUDA resident</th><th>CPU / CUDA + transfers</th><th>CPU / resident</th>' for _ in sizes)+'</tr></thead><tbody>'
    for name,items in sorted(by_name.items()):
        table+=f'<tr data-name="{name.lower()}"><th>{name}</th>'
        for size in sizes:
            row=items.get(size,{})
            for mode in ('cpu','cuda_host','cuda_resident'):
                value=row.get('timings',{}).get(mode,{}).get('p50_ms')
                label=f'{value:.3f}' if value is not None else '—'
                table+=f'<td><a href="#{size}-{name}">{label}</a></td>'
            timings=row.get('timings',{})
            for mode in ('cuda_host','cuda_resident'):
                cpu=timings.get('cpu',{}).get('p50_ms')
                gpu=timings.get(mode,{}).get('p50_ms')
                if cpu is None or gpu is None or gpu<=0:
                    table+='<td>—</td>'
                    continue
                ratio=cpu/gpu
                table+=f'<td style="color:{"#82dfb2" if ratio>=1 else "#ffb996"}">{ratio:.1f}×</td>'
        table+='</tr>'
    table+='</tbody></table></div>'
    page='''<!doctype html><html><meta charset="utf-8"><title>Full augmentation benchmark · Paris4</title>
<style>body{background:#101827;color:#e5edf7;font:16px system-ui;margin:30px auto;max-width:1240px;padding:0 20px}h1{font-size:32px}small{color:#9db0ca;font-size:17px}input{padding:14px;font-size:18px;width:95%;background:#1c2940;color:white;border:1px solid #617493;border-radius:8px}article{background:#162238;border-radius:12px;padding:20px;margin:26px 0}img{width:100%;height:auto}.params{font-family:monospace;overflow-wrap:anywhere}a{color:#90c5ff}p{line-height:1.5}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #34435c}th:first-child{text-align:left}thead{background:#1c2940}</style>
<h1>Full augmentation catalog · Paris4 scale 2</h1><p>128³ and 256³ float32 inputs, one common crop center. Five measured repetitions after one warmup. File reads, input preparation, validation and image rendering are excluded. CPU uses one thread; GPU calls are synchronized. Slices are selected from the changed regions, with coordinates labeled; original and augmented views use the same coordinates when their grids match. The preview uses the most visibly changed measured seed. Click any image to open its PNG.</p>
<p><b>CUDA + transfers</b> includes image upload and output download. <b>CUDA resident</b> uses images already on the GPU. Speedup columns divide CPU time by the corresponding CUDA time; above 1× means CUDA is faster.</p>
<p>Derived masks are CT &gt; 0.5, not ground-truth labels. The 2D-only tracing operations have separate companion results. Derived quantities show their own labeled display window. Timings, errors, settings and build metadata are in the JSON reports.</p>
<p><a href="timings.csv">Download timing table (CSV)</a> · <a href="128.json">128³ raw results</a> · <a href="256.json">256³ raw results</a></p>
<input id="search" placeholder="Filter augmentations by name…"><main>'''+table+''.join(rows)+'''</main><script>search.oninput=()=>{for(const item of document.querySelectorAll('[data-name]'))item.hidden=!item.dataset.name.includes(search.value.toLowerCase())}</script></html>'''
    scope='Utility transforms included.' if include_utilities else 'Utility transforms omitted; normalization included. Raw reports may contain earlier utility measurements.'
    page=page.replace('Full augmentation','Augmentation')
    page=page.replace('Augmentation benchmark · Paris4','Augmentation benchmark')
    page=page.replace('Augmentation catalog · Paris4 scale 2','Augmentation catalog')
    protocol='; '.join(f"{r['size']}³: {r.get('reps','?')} measured repetitions, {r.get('warmup','?')} warmups" for r in reports)
    page=page.replace('128³ and 256³ float32 inputs, one common crop center. Five measured repetitions after one warmup.',html.escape(protocol)+'. Float32 inputs.')
    links=' · '.join(f'<a href="{size}.json">{size}³ raw results</a>' for size in sizes)
    page=page.replace('<a href="128.json">128³ raw results</a> · <a href="256.json">256³ raw results</a>',links)
    sources=''.join(f'<p class="params">{r["size"]}³ source: {html.escape(str(r.get("array","unspecified")))} · start ZYX: {html.escape(str(r.get("start","unspecified")))}</p>' for r in reports)
    page=page.replace('<input id="search"',sources+'<input id="search"')
    page=page.replace('<input id="search"',f'<p>{scope}</p><input id="search"')
    extra=[]
    for report in reports:
        for run in report.get('supplementary_runs',[]):
            link=html.escape(run['report'],quote=True)
            extra.append(f'<a href="{link}">{report["size"]}³ supplementary measurements</a>')
    for link in sorted({r['comparison_report'] for r in reports if r.get('comparison_report')}):
        extra.append(f'<a href="{html.escape(link,quote=True)}">CPU optimization before/after</a>')
    for filename,label in [('before-after.csv','Before/after timings'),('verification.json','Validation results'),('profile-summary.json','Profiler check')]:
        if (directory/filename).exists():extra.append(f'<a href="{filename}">{label}</a>')
    if extra:page=page.replace('<input id="search"','<p>'+ ' · '.join(extra)+'</p><input id="search"')
    (directory/'index.html').write_text(page)
    header='size,augmentation,CPU_p50_ms,CUDA_host_p50_ms,CUDA_resident_p50_ms,status\n'
    import csv,io
    out=io.StringIO();out.write(header);writer=csv.writer(out)
    for report in reports:
        for row in report['results']:
            writer.writerow([report['size'],row['name'],*[row.get('timings',{}).get(m,{}).get('p50_ms','') for m in ('cpu','cuda_host','cuda_resident')],row.get('error','ok')])
    (directory/'timings.csv').write_text(out.getvalue())
