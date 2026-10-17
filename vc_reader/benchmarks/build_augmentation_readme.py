"""Build the main README and portable assets from recorded 128³ benchmarks.

No transforms or benchmarks are run. Existing numerical previews are exported
as original/augmented axis pairs. The saved docs snapshot can regenerate the
README in a fresh checkout without the original benchmark output directory.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
from PIL import Image

from augmentation_descriptions import DESCRIPTIONS
from augment_catalog import UTILITY_TRANSFORMS
from augment_preview import export_axis_previews

ROOT=Path(__file__).resolve().parents[1]


def parameter_names():
    tree=ast.parse((ROOT/'src/vc_reader/augment/__init__.pyi').read_text())
    return {node.name:[arg.arg for arg in node.args.args] for node in tree.body if isinstance(node,ast.FunctionDef)}


def named_recipe(parameters,names):
    call=ast.parse(parameters,mode='eval').body
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name):raise ValueError('expected a transform constructor')
    name=call.func.id
    keywords=[]
    for index,value in enumerate(call.args):
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='reference_volume':value=ast.Name(id='reference',ctx=ast.Load())
        keywords.append(ast.keyword(arg=names[name][index],value=value))
    call.args=[];call.keywords=keywords+call.keywords
    call.func=ast.Attribute(value=ast.Name(id='A',ctx=ast.Load()),attr=name,ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(call))


def copy_image(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()==target.resolve():return
    if target.suffix=='.webp' and source.suffix!='.webp':
        with Image.open(source) as image:image.save(target,lossless=True,method=6)
    else:shutil.copyfile(source,target)


def timing_table(row):
    values=[row['timings'][mode]['p50_ms'] for mode in ('cpu','cuda_host','cuda_resident')]
    cpu,uploaded,resident=values
    return [
        '| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |',
        '| ---: | ---: | ---: | ---: | ---: |',
        f'| {cpu:.3f} ms | {uploaded:.3f} ms | {resident:.3f} ms | {cpu/uploaded:.1f}× | {cpu/resident:.1f}× |',
    ]


def render_readme(report,asset_prefix):
    rows=report['results'];volume=[r for r in rows if r['sample'] not in ('strip','surface')]
    companion=[r for r in rows if r['sample'] in ('strip','surface')]
    center=tuple(x+report['size']//2 for x in report['start'])
    lines=[
        '# vc_reader','', '## Augmentations','',
        f'{len(volume)} volumetric augmentation examples, plus {len(companion)} 2D tracing companions, with CPU and CUDA implementations. Normalization is included; utility transforms are omitted.', '',
        'Each entry shows the **measured configuration**, not the constructor defaults. Use `from vc_reader import augment as A`; see the [augmentation API guide](docs/augmentation.md) for options and pipeline usage.', '',
        f'Benchmarks use **128 × 128 × 128 float32** crops from PHercParis4 volume `20260411134726-2.400um-0.2m-78keV-masked.zarr`, **scale 2**, centered at ZYX `{center}`. CT values are divided by 255. Timings are medians of {report["reps"]} measured calls after {report["warmup"]} warmup, using one CPU thread and a Release build.', '',
        f'Hardware: **{report["cpu"]}** and **{report["gpu"].split(",")[0]}**. Calls are synchronous; file reads, setup, output comparison and image rendering are excluded. Public-call copies and input validation remain included.', '',
        '**CUDA uploaded** includes uploading the host input and downloading the output. **CUDA resident** starts and ends with device arrays. Speedup is CPU time divided by the corresponding CUDA time; values above 1× mean CUDA is faster.', '',
        '**Images:** Z, Y and X slice pairs show the original above the augmented output. Planes are selected where the effect is visible, with the same display window for each pair. Coordinates and windows are printed on the images; gold boxes mark small changed regions. Crop/resize examples use corresponding relative grid positions. Click an image for full resolution.', '',
        'Most examples use one CT channel. Channel misalignment uses two; label operations use a foreground mask derived from CT > 0.5. `reference` denotes a second 128³ CT crop, 384 scale-2 voxels farther along Z. These masks and the 2D coordinate surface are benchmark inputs, not manual annotations.', '',
        f'[Recorded timings, parameters, seeds and build metadata]({asset_prefix}/benchmark-128.json) · [Benchmark and regeneration commands](benchmarks/README.md#main-readme-augmentation-gallery)', '',
        'Expand an augmentation below to see its parameters, timings and images. Elastic also includes a WebM animation.', '',
    ]
    def entry(row,is_volume):
        name=row['name']
        lines.extend(['<details>',f'<summary><strong>{name}</strong></summary>','',row['description'],'','```python',row['recipe'],'```',''])
        lines.extend(timing_table(row));lines.append('')
        if row.get('animation'):
            animation=row['animation'];video=f'{asset_prefix}/{animation["video"]}'
            poster=f'{asset_prefix}/{animation["poster"]}'
            lines.extend([f'[▶ Watch {name} animation (WebM)]({video})','',animation['caption'],'',
                          f'[![{name} animation poster]({poster})]({video})',''])
        if is_volume:
            lines.extend(['| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |','| :---: | :---: | :---: |'])
            cells=[]
            for axis,path in zip(('Z','Y','X'),row['axis_previews']):
                url=f'{asset_prefix}/{path}'
                cells.append(f'[![{name}: original and augmented {axis} slices]({url})]({url})')
            lines.extend(['| '+' | '.join(cells)+' |',''])
        else:
            url=f'{asset_prefix}/{row["preview"]}'
            lines.extend([f'[![{name}: original and augmented 2D views]({url})]({url})',''])
        if row.get('note'):lines.extend([row['note'],''])
        lines.extend(['</details>',''])
    for row in volume:entry(row,True)
    if companion:
        lines.extend(['### 2D tracing companions','',
                      'These operate on 128 × 128 inputs derived from the 128³ crop. Their timings are for 2D work and they have no Z/Y/X volume slice triplet.',''])
        for row in companion:entry(row,False)
    return '\n'.join(lines)


def build(report_path,readme,assets):
    report=json.loads(report_path.read_text())
    if report['size']!=128:raise ValueError('README examples require the recorded 128³ benchmark')
    selected=sorted((r for r in report['results'] if r['name'] not in UTILITY_TRANSFORMS),key=lambda r:r['name'].lower())
    names=parameter_names()
    for row in selected:
        if row['name'] not in DESCRIPTIONS:raise ValueError('missing description: '+row['name'])
        if row.get('error'):raise ValueError('failed benchmark: '+row['name'])
        for mode in ('cpu','cuda_host','cuda_resident'):
            if row['timings'][mode]['p50_ms']<=0:raise ValueError('invalid timing: '+row['name'])
    metadata={k:v for k,v in report.items() if k not in ('results','supplementary_runs','comparison_report')}
    runs=dict(metadata.get('runs',{}))
    if not runs:
        runs['original']={k:v for k,v in metadata.items() if k!='runs'}
        for run in report.get('supplementary_runs',[]):
            value=json.loads((report_path.parent/run['report']).read_text())
            runs[run['report']]={k:v for k,v in value.items() if k!='results'}
    metadata['runs']=runs;metadata['results']=[]
    for original in selected:
        row=dict(original);name=row['name']
        row['description']=DESCRIPTIONS[name];row['recipe']=named_recipe(row['parameters'],names)
        if row['sample'] not in ('strip','surface'):
            if row.get('axis_previews'):
                paths=[]
                for source in row['axis_previews']:
                    target=assets/'previews'/Path(source).with_suffix('.webp').name
                    copy_image(report_path.parent/source,target);paths.append(target)
            else:
                paths=export_axis_previews(report_path.parent/row['preview'],assets/'previews',name)
            row['axis_previews']=[path.relative_to(assets).as_posix() for path in paths]
            row.pop('preview',None)
        else:
            target=assets/'previews'/f'{name}.webp'
            copy_image(report_path.parent/row['preview'],target)
            row['preview']=target.relative_to(assets).as_posix()
        if row.get('animation'):
            animation=dict(row['animation'])
            for key in ('video','poster','manifest'):
                source=Path(animation[key]);filename=source.with_suffix('.webp').name if key=='poster' else source.name
                target=assets/'animations'/filename
                copy_image(report_path.parent/source,target)
                animation[key]=target.relative_to(assets).as_posix()
            row['animation']=animation
        row['run_id']=row.pop('benchmark_report',row.get('run_id','original'))
        metadata['results'].append(row)
    assets.mkdir(parents=True,exist_ok=True)
    (assets/'benchmark-128.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prefix=assets.relative_to(readme.parent).as_posix()
    readme.write_text(render_readme(metadata,prefix))
    print(f'Wrote {readme}: {len(selected)} entries; assets in {assets}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'docs/augmentations/benchmark-128.json')
    parser.add_argument('--readme',type=Path,default=ROOT/'README.md')
    parser.add_argument('--assets',type=Path,default=ROOT/'docs/augmentations')
    args=parser.parse_args()
    build(args.report.resolve(),args.readme.resolve(),args.assets.resolve())


if __name__=='__main__':main()
