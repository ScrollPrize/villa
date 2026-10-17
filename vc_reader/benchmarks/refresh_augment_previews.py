"""Regenerate catalog PNGs from recorded inputs/seeds without rerunning timings."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from vc_reader import augment as A
from vc_reader.reader import ZarrReader
from augment_catalog import catalog, inputs
from augment_preview import effect_strength, preview, gallery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--localized-only',action='store_true',help='Refresh only previews with sparse changes; retain the other images.')
    parser.add_argument('--include-utilities',action='store_true',help='Also run and display utility transforms.')
    parser.add_argument('--html-only',action='store_true',help='Rebuild HTML and timing CSV from saved results without reading volumes or running transforms.')
    parser.add_argument('--only',default='',help='Refresh only these comma-separated transform names; retain other previews.')
    args=parser.parse_args()
    selected={name.strip() for name in args.only.split(',') if name.strip()}
    reports=[]
    for path in sorted(args.directory.glob('[0-9]*.json')):
        report=json.loads(path.read_text());size=report['size']
        reports.append(report)
        if args.html_only:continue
        with ZarrReader.open(report['array'],threads=1) as reader:
            image=reader.read(report['start'],(size,)*3).astype(np.float32)[None]/np.float32(255)
            reference=reader.read(report['reference_start'],(size,)*3).astype(np.float32)[None]/np.float32(255)
        sources=inputs(image);cases={case.name:case for case in catalog(size,reference,include_utilities=args.include_utilities)}
        for row in report['results']:
            if selected and row['name'] not in selected:continue
            if row['name'] not in cases or not row.get('preview'):continue
            counts=row.get('preview_selection',{}).get('changed_voxels',[])
            if args.localized_only and not any(0<n<size*size/4 for n in counts):continue
            case=cases[row['name']];source=sources[case.sample]
            if case.name=='Elastic':
                # Older reports use a weaker Elastic recipe. Keep their images
                # tied to the measured parameters when the catalog changes.
                call=ast.parse(row['parameters'],mode='eval').body
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Elastic':
                    raise ValueError('expected a recorded Elastic constructor')
                case.transforms=[A.Elastic(*(ast.literal_eval(arg) for arg in call.args),
                                          **{kw.arg:ast.literal_eval(kw.value) for kw in call.keywords})]
            pipeline=A.Pipeline(case.transforms,targets=dict(image=A.Image(),mask=A.Label(),surface=A.CoordinateField()),backend='cpu')
            seed=row.get('preview_selection',{}).get('seed')
            if seed is None:
                choices=[(123+i,pipeline(source,seed=123+i)) for i in range(report['reps'])]
                seed,result=max(choices,key=lambda item:effect_strength(case,source,item[1]))
            else:result=pipeline(source,seed=seed)
            row['preview_selection']=preview(args.directory/row['preview'],case,source,result,size,row['parameters'],seed)
            print(size,case.name,row['preview_selection'],flush=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    gallery(args.directory,reports,include_utilities=args.include_utilities)


if __name__=='__main__':main()
