"""Generate the README's transform WebPs from a real Zarr crop.

Uses the explicit benchmark cases in transform_cases.py.
Requires vc-reader[reader,transforms] and Pillow with WebP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from transform_cases import CASES, input_sample

ROOT = Path(__file__).resolve().parents[1]


PREVIEW_COLUMNS = 3
PREVIEW_TILE = 360
PREVIEW_GAP = 24
PREVIEW_TOP = 210
PREVIEW_ROW_HEIGHT = PREVIEW_TILE + 62
PREVIEW_FOOTER = 65


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def preview_volumes(case, source, result):
    output=result[case.output]
    key='mask' if case.sample=='labels' else ('surface' if case.sample=='surface' else 'image')
    original=source[key]
    if case.name=='ToPseudo2D':output=output.reshape(original.shape)
    return key, original, output, original[min(case.channel,original.shape[0]-1)], output[min(case.channel,output.shape[0]-1)]


def effect_strength(case, source, result):
    _,_,_,a,b=preview_volumes(case,source,result)
    if a.shape!=b.shape:return float('inf')
    return int(np.count_nonzero(a!=b))


def effect_slices(original, transformed, *, derived=False, interior_fraction=0):
    """Maximize changed area per axis; ties favor the center, never an empty slice.

    Changed grids are compared at corresponding relative positions, not claimed
    to be registered. Derived outputs select planes with the most output signal.
    An interior search excludes boundary padding from both scoring and selection.
    """
    mapped=original
    if original.shape!=transformed.shape:
        for axis,n in enumerate(transformed.shape):
            mapped=np.take(mapped,np.rint(np.linspace(0,mapped.shape[axis]-1,n)).astype(int),axis=axis)
    difference=np.abs(transformed.astype(np.float32)-mapped.astype(np.float32))
    signal=np.abs(transformed.astype(np.float32)) if derived else difference
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
    original_indices=[round(i*(original.shape[a]-1)/max(1,transformed.shape[a]-1)) for a,i in enumerate(indices)]
    return original_indices,indices,difference


def changed_bounds(delta):
    """Mark small changed regions without obscuring broadly changed images."""
    rows,cols=np.nonzero(delta)
    if not len(rows):return None
    y0,y1=int(rows.min()),int(rows.max())+1
    x0,x1=int(cols.min()),int(cols.max())+1
    if (y1-y0)*(x1-x0)>delta.size/4:return None
    return max(0,x0-3),max(0,y0-3),min(delta.shape[1]-1,x1+3),min(delta.shape[0]-1,y1+3)


def preview(case, source, result, size, parameters, seed):
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
        panels += [(f'Transformed {case.output} · {axis}={index}',np.take(b,index,axis=j),True)
                   for j,(axis,index) in enumerate(zip(('Z','Y','X'),after))]
        if a.shape==b.shape and not derived:
            for axis,index in enumerate(after):
                box=changed_bounds(np.take(delta,index,axis=axis))
                if box is not None:highlights[axis]=box
        metadata=dict(original_indices=before,transformed_indices=after,
                      changed_voxels=[int(np.count_nonzero(np.take(delta,index,axis=j))) for j,index in enumerate(after)],
                      selection=selection,seed=seed)
        if interior_fraction:metadata['interior_fraction']=interior_fraction
    else:
        panels=[('Original · central Z plane / 2D input',a,False),('Transformed · 2D output',b,True)]
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
    return canvas, metadata


def generate(args):
    from vc_reader import transforms as A, _transforms
    from vc_reader.reader import ZarrReader

    if not features.check("webp"):
        raise ValueError("Pillow was built without WebP support")
    selected = {case.name: case for case in CASES}
    if args.only:
        missing = set(args.only) - selected.keys()
        if missing:
            raise ValueError("unknown transform names: " + ", ".join(sorted(missing)))
        selected = {name: case for name, case in selected.items() if name in args.only}
    with ZarrReader.open(args.array, level=args.level, threads=args.threads) as reader:
        if reader.ndim != 3:
            raise ValueError("input must be a 3-D Zarr array")
        shape = (args.size,) * 3
        start = args.start or [(n - args.size) // 2 for n in reader.shape]
        # Keep the reference distinct when possible: +384 Z voxels for 128-cubed inputs.
        reference_start = args.reference_start or [min(start[0] + 3 * args.size, reader.shape[0] - args.size), *start[1:]]
        for origin in (start, reference_start):
            if any(s < 0 or s + args.size > n for s, n in zip(origin, reader.shape)):
                raise ValueError("input/reference crop must fit inside the volume")
        image = reader.read_region(start, shape, dtype=np.float32)[None] / np.float32(args.divisor)
        reference = reader.read_region(reference_start, shape, dtype=np.float32)[None] / np.float32(args.divisor)
        source_metadata = dict(shape=list(reader.shape), chunks=list(reader.chunks), dtype=str(reader.dtype))
    if not np.isfinite(image).all() or not np.isfinite(reference).all() or np.ptp(image) == 0:
        raise ValueError("choose a finite crop with visible CT signal (not constant background)")
    manifest = dict(array=args.array, level=args.level, source=source_metadata, start=start, reference_start=reference_start,
                    size=args.size, divisor=args.divisor, seed=args.seed, seed_trials=args.seed_trials,
                    cases_sha256=hashlib.sha256(Path(__file__).with_name('transform_cases.py').read_bytes()).hexdigest(),
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                    extension_sha256=hashlib.sha256(Path(_transforms.__file__).read_bytes()).hexdigest(), previews={})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, case in selected.items():
        recipe = case.recipe()
        transform = case.construct(A, reference)
        pipeline = A.Pipeline([transform], targets=dict(image=A.Image(), mask=A.Label(), surface=A.CoordinateField()), backend="cpu")
        source = input_sample(image, case.sample)
        best_score = -1
        for seed in range(args.seed, args.seed + args.seed_trials):
            result = pipeline(source, seed=seed)
            if not all(np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray)):
                raise ValueError(f"{name} produced non-finite values")
            score = effect_strength(case, source, result)
            if score > best_score:
                best_score, best_seed, best_result = score, seed, result
        canvas, metadata = preview(case, source, best_result, args.size, recipe, best_seed)
        paths = []
        if case.sample in ("strip", "surface"):
            path = args.output / f"{name}.webp"
            canvas.save(path, format="WEBP", lossless=True, method=6)
            paths.append(path.name)
        else:
            for axis, label in enumerate(("z", "y", "x")):
                left = PREVIEW_GAP + axis * (PREVIEW_TILE + PREVIEW_GAP)
                panel = canvas.crop((left, PREVIEW_TOP, left + PREVIEW_TILE, PREVIEW_TOP + 2 * PREVIEW_ROW_HEIGHT))
                path = args.output / f"{name}-{label}.webp"
                panel.save(path, format="WEBP", lossless=True, method=6)
                paths.append(path.name)
        manifest["previews"][name] = dict(recipe=recipe, files=paths, note=case.note, **metadata)
        print(f"{name}: seed={best_seed}, " + ", ".join(paths), flush=True)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {sum(len(row['files']) for row in manifest['previews'].values())} WebPs to {args.output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", required=True, help="real local/HTTP/S3 Zarr v2 array or group")
    parser.add_argument("--level", help="array level within a Zarr group")
    parser.add_argument("--start", type=int, nargs=3, metavar=("Z", "Y", "X"), help="crop origin; default: volume center")
    parser.add_argument("--reference-start", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--size", type=int, default=128, help="cube edge; cases retain their explicit sizes")
    parser.add_argument("--divisor", type=float, default=255, help="divide CT intensities by this value; use 1 for already normalized data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seed-trials", type=int, default=5, help="select the most visibly changed seed deterministically")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--only", nargs="+", help="subset of transform case names")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/transforms/previews")
    parser.add_argument("--manifest", type=Path, help="optional JSON provenance: recipes, inputs, seeds, planes and hashes")
    args = parser.parse_args()
    if min(args.size, args.seed_trials, args.threads) < 1 or args.seed < 0:
        parser.error("size, seed-trials and threads must be positive; seed must be nonnegative")
    if not np.isfinite(args.divisor) or args.divisor <= 0:
        parser.error("divisor must be finite and positive")
    try:
        generate(args)
    except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Preview generation failed: {exc}") from exc


if __name__ == "__main__":
    main()


