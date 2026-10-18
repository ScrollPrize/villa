"""Real geometry/volume surface acceptance and six-request performance protocol.

Defaults to the bundled PHercParis4 geometry and its existing local volume.
For projection supply serialized, explicitly registered real label/reference
Sources; an ink label's render relationship alone does not establish its UV map.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from bench_sampling import timed


def host(array):
    return array.numpy() if hasattr(array,'numpy') else array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','cuda','both'),default='both')
    parser.add_argument('--label-source',type=Path)
    parser.add_argument('--reference-source',type=Path)
    parser.add_argument('--start',type=int,nargs=2)
    parser.add_argument('--size',type=int,nargs=2,default=(32,32))
    parser.add_argument('--cache-dir',default='/tmp/vc-reader-surface-cache')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from bench_pipeline import load_extensions
    load_extensions(args.native_dir)
    from vc_reader import Source,Storage,Surface,Affine,VolumeRegistration
    from vc_reader.sampling import open_sample
    from vc_reader.surface import SurfaceGeometry
    from scipy.ndimage import map_coordinates
    import tifffile
    path=Path(__file__).parent/'fixtures/fiber_132487'
    if args.label_source:
        if not args.reference_source or not args.start:
            parser.error('projection requires --reference-source and --start')
        label=Source.from_dict(json.loads(args.label_source.read_text()))
        reference=Source.from_dict(json.loads(args.reference_source.read_text()))
        surface=label.registration.surface
        targets={'ink':label}
        requests=[((args.start[0]+i*2,args.start[1]+i*2),tuple(args.size)) for i in range(6)]
    else:
        reference=Source('PHercParis4/20260411134726/2',Storage('/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2','zarr'),role='image',axes=('z','y','x'),
                         registration=VolumeRegistration('PHercParis4','20260411134726',affine=Affine.scale(4),volume_spacing_um=(2.4,)*3))
        surface=Surface('fiber_132487',Storage(str(path),'tifxyz'),'PHercParis4','20260411134726',axes=('y','x'),component_order=('x','y','z'),xyz_to_volume=Affine.scale(4),spacing_um=(9.6,)*3)
        targets={}
        requests=[((30,x),tuple(args.size)) for x in (1000,2000,3000,4000,5000,6000)]
    options=dict(cache_dir=args.cache_dir,offline=args.offline,threads=1,memory_cache=128<<20,sampling_cache_bytes=128<<20)
    report=dict(command=sys.argv,reference=reference.to_dict(),label=label.to_dict() if args.label_source else None,
                requests=requests,protocol=dict(warmups=2,repetitions=10,requests=6,os_cache_cleared=False),runs={})
    baseline={}
    for backend in ('cpu','cuda') if args.backend=='both' else (args.backend,):
        now=time.perf_counter()
        with open_sample(reference,targets,backend=backend,**options) as reader:
            opened=(time.perf_counter()-now)*1000
            def operation(start,size):
                return reader.project_surface(surface,start,size,label_target='ink',thickness_um=2.4,padding_um=2.4) if targets else reader.render_surface(surface,start,size,offsets_um=(-9.6,0,9.6))
            first=[];accepted=[]
            for i,request in enumerate(requests):
                now=time.perf_counter();result=operation(*request);first.append((time.perf_counter()-now)*1000)
                actual={k:host(t.data) for k,t in result.targets.items()}
                valid={k:host(t.valid) for k,t in result.targets.items()}
                masks={k:{m:host(v) for m,v in t.masks.items()} for k,t in result.targets.items()}
                if backend=='cpu':baseline[i]=(actual,valid,masks)
                elif i in baseline:
                    for k in actual:
                        if actual[k].dtype.kind in 'bui':np.testing.assert_array_equal(actual[k],baseline[i][0][k])
                        else:np.testing.assert_allclose(actual[k],baseline[i][0][k],atol=1e-3,rtol=2e-5)
                        np.testing.assert_array_equal(valid[k],baseline[i][1][k])
                    for k,group in masks.items():
                        for m,value in group.items():
                            np.testing.assert_array_equal(value,baseline[i][2][k][m])
                if targets:
                    m=result.metadata['projection']
                    with reference.open(cache_dir=args.cache_dir,offline=args.offline) as raw:
                        expected=raw.read(tuple(m['origin_zyx']),tuple(m['shape_zyx']))
                    np.testing.assert_array_equal(actual['image'][0],expected)
                    assert valid['ink'].any()
                    # Independently rasterize the first real patch at zero thickness.
                    # TIFF geometry is read directly; the oracle uses double precision
                    # and its own clipping, intersection and closest-point routines.
                    if i == 0:
                        from surface_reference import crop_triangles, label_sampler, project
                        from vc_reader.reader.encoded_files import EncodedFiles
                        from vc_reader.reader.source_resolution import pixel_affine
                        root=surface.geometry.location.rstrip('/')
                        files=EncodedFiles(cache_dir=args.cache_dir,offline=args.offline)
                        with files.open(root+'/meta.json') as (handle,_):meta=json.load(handle)
                        files.close()
                        scale=meta['scale'];to_uv=Affine.from_rows(((1/scale[0],0,0,0),(0,1/scale[1],0,0),(0,0,1,0)))
                        label_to_uv=pixel_affine(label.registration.pixel_to_uv)
                        from contextlib import ExitStack
                        placement=reference.registration.affine.inverse()@surface.xyz_to_volume
                        zero=reader.project_surface(surface,*request,label_target='ink')
                        zmeta=zero.metadata['projection']
                        with ExitStack() as stack:
                            fields=[stack.enter_context(Source('oracle/'+axis,Storage(root+'/'+axis+'.tif','tiff'),
                                    axes=('y','x'),role='scalar').open(cache_dir=args.cache_dir,offline=args.offline)) for axis in 'xyz']
                            raw=stack.enter_context(label.open(cache_dir=args.cache_dir,offline=args.offline))
                            triangles=crop_triangles(fields,placement,label_to_uv.inverse()@to_uv,
                                                     zmeta['origin_zyx'],zmeta['shape_zyx'])
                            oracle=project(triangles,label_sampler(raw),raw.dtype,zmeta['origin_zyx'],zmeta['shape_zyx'],surface.spacing_um)
                        for name,actual_field,expected_field in zip(('values','valid','conflict','coverage'),
                                (zero['ink'].data,zero['ink'].valid,zero['ink'].masks['conflict'],zero['ink'].masks['coverage']),
                                (*oracle[:3],oracle[4])):
                            np.testing.assert_array_equal(host(actual_field)[0],expected_field,err_msg=name)
                        report['independent_projection_oracle']='zero thickness: values, coverage and conflicts match float64 scalar geometry'
                else:
                    # Independent TIFF + SciPy coordinate interpolation at zero offset.
                    start,size=request;y,x=np.indices(size,dtype=np.float64)
                    coord=[(y+start[0])*.2,(x+start[1])*.25]
                    xyz=np.stack([map_coordinates(tifffile.imread(path/(a+'.tif')).astype(np.float64),coord,order=1,prefilter=False) for a in 'xyz'],axis=-1)
                    with SurfaceGeometry(surface) as geom:
                        points,support,origin=geom.read_points(start,size)
                    error=float(np.max(np.abs((points[0]+np.array(origin)-xyz)[support[0]])))
                    assert error<1e-4,error
                    with reference.open() as raw:
                        lo=np.floor(xyz.reshape(-1,3).min(axis=0)).astype(int)[::-1]
                        hi=np.ceil(xyz.reshape(-1,3).max(axis=0)).astype(int)[::-1]+1
                        image=raw.read(tuple(lo),tuple(hi-lo))
                    expected=map_coordinates(image.astype(np.float32),np.moveaxis(xyz[...,::-1]-lo,-1,0),order=1,prefilter=False)
                    np.testing.assert_allclose(actual['image'][0,1][support[0]],expected[support[0]],atol=.02,rtol=2e-5)
                accepted.append(dict(metadata=result.metadata,hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in actual.items()},covered={k:int(v.sum()) for k,v in valid.items()}))
            stats=reader.stats()
            run=timed(operation,requests)
            run.update(open_ms=opened,first_requests_ms=first,initial_stats=stats,final_stats=reader.stats(),acceptance=accepted)
            report['runs'][backend]=run
            print(backend,run['summary'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=='__main__':main()
