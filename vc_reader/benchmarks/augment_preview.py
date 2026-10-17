"""Render numeric orthogonal slices and a self-contained benchmark index."""
import html
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bench_augment_native import arrays


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


def effect_slices(original, augmented, *, derived=False):
    """Maximize changed area per axis; ties favor the center, never an empty slice.

    Changed grids are compared at corresponding relative positions, not claimed
    to be registered. Derived outputs select planes with the most output signal.
    """
    mapped=original
    if original.shape!=augmented.shape:
        for axis,n in enumerate(augmented.shape):
            mapped=np.take(mapped,np.rint(np.linspace(0,mapped.shape[axis]-1,n)).astype(int),axis=axis)
    difference=np.abs(augmented.astype(np.float32)-mapped.astype(np.float32))
    signal=np.abs(augmented.astype(np.float32)) if derived else difference
    indices=[]
    for axis in range(3):
        other=tuple(a for a in range(3) if a!=axis)
        scores=np.count_nonzero(signal,axis=other)
        strongest=scores.max()
        candidates=np.flatnonzero(scores==strongest)
        if strongest:
            magnitudes=signal.sum(axis=other)
            candidates=candidates[magnitudes[candidates]==magnitudes[candidates].max()]
        indices.append(int(candidates[np.argmin(np.abs(candidates-(augmented.shape[axis]-1)/2))]))
    original_indices=[round(i*(original.shape[a]-1)/max(1,augmented.shape[a]-1)) for a,i in enumerate(indices)]
    return original_indices,indices,difference


def preview(path, case, source, result, size, parameters, seed):
    key,original,output,a,b=preview_volumes(case,source,result)
    derived=case.name in ('DistanceTransform','StructureTensor','SurfacePerturb')
    panels=[]
    selection='2D input'; metadata={}
    if a.ndim==3 and b.ndim==3:
        before,after,delta=effect_slices(a,b,derived=derived)
        selection='most affected slices' if np.any(delta) else 'no voxel-value change for this sample'
        if derived:selection='slices with strongest derived output'
        if a.shape!=b.shape:selection+=' · coordinates mapped by relative grid position'
        panels=[(f'Original {key} · {axis}={index}',np.take(a,index,axis=j),False)
                for j,(axis,index) in enumerate(zip(('Z','Y','X'),before))]
        panels += [(f'Augmented {case.output} · {axis}={index}',np.take(b,index,axis=j),True)
                   for j,(axis,index) in enumerate(zip(('Z','Y','X'),after))]
        metadata=dict(original_indices=before,augmented_indices=after,
                      changed_voxels=[int(np.count_nonzero(np.take(delta,index,axis=j))) for j,index in enumerate(after)],
                      selection=selection,seed=seed)
    else:
        panels=[('Original · central Z plane / 2D input',a,False),('Augmented · 2D output',b,True)]
        if case.name=='SurfacePerturb':
            delta=np.sqrt(np.sum((output-original)**2,axis=0))
            panels.append(('Coordinate displacement magnitude',delta,True))
    joined=[p for _,p,_ in panels]
    lo=min(float(v.min()) for v in joined);hi=max(float(v.max()) for v in joined)
    if not derived and case.sample in ('image','both','signed','channels','strip') and case.name not in ('Normalize',):
        lo=min(0.,lo);hi=max(1.,hi)
    widths=3;tile=360;gap=24;top=210;rows=(len(panels)+2)//3
    canvas=Image.new('RGB',(widths*tile+(widths+1)*gap,top+rows*(tile+62)+65),'#101827')
    draw=ImageDraw.Draw(canvas)
    draw.text((gap,20),f'{case.name}  |  {size}³ input'+(' · 2D companion' if a.ndim==2 else ''),font=font(29),fill='white')
    y=65
    for line in textwrap.wrap(parameters,110)[:4]:
        draw.text((gap,y),line,font=font(16),fill='#c6d3e5');y+=23
    draw.text((gap,top-30),f'CPU output · seed {seed} · channel {case.channel} · {selection}',font=font(15),fill='#a4b5ce')
    for i,(title,plane,is_output) in enumerate(panels):
        x=gap+(i%widths)*(tile+gap);y=top+(i//widths)*(tile+62)
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
        canvas.paste(picture,(x+(tile-picture.width)//2,y+29+(tile-picture.height)//2))
        draw.text((x,y+tile+33),f'window [{low:.4g}, {high:.4g}] · {plane.shape[0]} × {plane.shape[1]}',font=font(14),fill='#a4b5ce')
    footer=case.note or ('Derived output uses different units; windows are labeled. No per-slice autoscaling.' if derived else 'Shared display window; no per-slice contrast adjustment. Z→YX, Y→ZX, X→ZY.')
    for i,line in enumerate(textwrap.wrap(footer,125)[:2]):
        draw.text((gap,canvas.height-54+i*22),line,font=font(15),fill='#a4b5ce')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return metadata


def gallery(directory, reports):
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
    table='<div style="overflow:auto"><table><thead><tr><th>Augmentation</th>'+''.join(f'<th colspan="3">{size}³ · median ms</th>' for size in sizes)+'</tr><tr><th></th>'+''.join('<th>CPU</th><th>CUDA + transfers</th><th>CUDA resident</th>' for _ in sizes)+'</tr></thead><tbody>'
    for name,items in sorted(by_name.items()):
        table+=f'<tr data-name="{name.lower()}"><th>{name}</th>'
        for size in sizes:
            row=items.get(size,{})
            for mode in ('cpu','cuda_host','cuda_resident'):
                value=row.get('timings',{}).get(mode,{}).get('p50_ms')
                label=f'{value:.3f}' if value is not None else '—'
                table+=f'<td><a href="#{size}-{name}">{label}</a></td>'
        table+='</tr>'
    table+='</tbody></table></div>'
    page='''<!doctype html><html><meta charset="utf-8"><title>Full augmentation benchmark · Paris4</title>
<style>body{background:#101827;color:#e5edf7;font:16px system-ui;margin:30px auto;max-width:1240px;padding:0 20px}h1{font-size:32px}small{color:#9db0ca;font-size:17px}input{padding:14px;font-size:18px;width:95%;background:#1c2940;color:white;border:1px solid #617493;border-radius:8px}article{background:#162238;border-radius:12px;padding:20px;margin:26px 0}img{width:100%;height:auto}.params{font-family:monospace;overflow-wrap:anywhere}a{color:#90c5ff}p{line-height:1.5}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #34435c}th:first-child{text-align:left}thead{background:#1c2940}</style>
<h1>Full augmentation catalog · Paris4 scale 2</h1><p>128³ and 256³ float32 inputs, one common crop center. Five measured repetitions after one warmup. File reads, input preparation, validation and image rendering are excluded. CPU uses one thread; GPU calls are synchronized. Slices are selected from the changed regions, with coordinates labeled; original and augmented views use the same coordinates when their grids match. The preview uses the most visibly changed measured seed. Click any image to open its PNG.</p>
<p>Derived masks are CT &gt; 0.5, not ground-truth labels. The 2D-only tracing operations have separate companion results. Derived quantities show their own labeled display window. Timings, errors, settings and build metadata are in the JSON reports.</p>
<p><a href="timings.csv">Download timing table (CSV)</a> · <a href="128.json">128³ raw results</a> · <a href="256.json">256³ raw results</a></p>
<input id="search" placeholder="Filter augmentations by name…"><main>'''+table+''.join(rows)+'''</main><script>search.oninput=()=>{for(const item of document.querySelectorAll('[data-name]'))item.hidden=!item.dataset.name.includes(search.value.toLowerCase())}</script></html>'''
    (directory/'index.html').write_text(page)
    header='size,augmentation,CPU_p50_ms,CUDA_host_p50_ms,CUDA_resident_p50_ms,status\n'
    import csv,io
    out=io.StringIO();out.write(header);writer=csv.writer(out)
    for report in reports:
        for row in report['results']:
            writer.writerow([report['size'],row['name'],*[row.get('timings',{}).get(m,{}).get('p50_ms','') for m in ('cpu','cuda_host','cuda_resident')],row.get('error','ok')])
    (directory/'timings.csv').write_text(out.getvalue())
