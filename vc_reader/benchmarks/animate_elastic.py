"""Render a fixed-field Elastic deformation as a three-axis WebM comparison.

Uses the public CPU pipeline's sampled displacement field and verifies the
trilinear display sampler against its endpoint. Requires existing Pillow,
SciPy and GStreamer (rawvideoparse, videoconvert, vp8enc, webmmux, vp8dec).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
from vc_reader import augment as A
from vc_reader.reader import ZarrReader
from augment_catalog import catalog,inputs
from augment_preview import font
from elastic_replay import displacement


def slice_samplers(image,field,indices):
    """Coordinates stay fixed; only the sampled production field is scaled."""
    # Replay coordinates are doubles in the native sampler; retain that precision
    # to verify its endpoint, but allocate only the three displayed planes.
    samplers=[]
    for axis,index in enumerate(indices):
        shape=tuple(n for a,n in enumerate(image.shape) if a!=axis)
        grid=iter(np.indices(shape,dtype=field.dtype))
        base=np.stack([np.full(shape,index,dtype=field.dtype) if a==axis else next(grid) for a in range(3)])
        samplers.append((base,np.moveaxis(np.take(field,index,axis=axis),-1,0)))
    return samplers


def sample_planes(image,samplers,fraction):
    return [map_coordinates(image,base+fraction*offset,order=1,mode='nearest',prefilter=False)
            for base,offset in samplers]


def animation_frame(original,planes,indices,fraction,rms,seed):
    tile,gap,header,row_height=320,16,112,352
    canvas=Image.new('RGB',(3*tile+4*gap,header+2*row_height+32),'#101827')
    draw=ImageDraw.Draw(canvas)
    draw.text((gap,10),'Elastic: cubic B-spline | sigma=32, magnitude=8, border=True',font=font(23),fill='white')
    draw.text((gap,43),f'Seed {seed} | deformation {fraction:.0%} | RMS {fraction*rms:.2f} voxels',font=font(20),fill='#c6d3e5')
    draw.text((gap,75),'Fixed interior slices | original above, deforming volume below | window [0, 1]',font=font(18),fill='#a4b5ce')
    for row,values in enumerate((original,planes)):
        for axis,plane in enumerate(values):
            x=gap+axis*(tile+gap);y=header+row*row_height
            draw.text((x,y),f'{"Original" if row==0 else "Deformed"} {"ZYX"[axis]}={indices[axis]}',font=font(20),fill='white')
            pixels=np.rint(np.clip(plane,0,1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((tile,tile),Image.Resampling.NEAREST),(x,y+28))
    draw.text((gap,canvas.height-25),'One displacement field scaled from 0 to 100% and back; no image crossfade.',font=font(17),fill='#a4b5ce')
    return canvas


def encode_webm(path,frames,order,fps):
    executable=shutil.which('gst-launch-1.0')
    if executable is None:raise RuntimeError('gst-launch-1.0 is required; no dependencies are installed automatically')
    width,height=frames[0].size
    command=[executable,'-q','fdsrc','fd=0','!','rawvideoparse','format=rgb',f'width={width}',f'height={height}',f'framerate={fps}/1',
             '!','videoconvert','!','video/x-raw,format=I420,colorimetry=bt601','!','vp8enc','deadline=1000000','cpu-used=4','threads=4',
             'end-usage=cq','cq-level=4','max-quantizer=4','target-bitrate=8000000',f'keyframe-max-dist={fps}',
             '!','webmmux','!','filesink',f'location={path}']
    with tempfile.TemporaryFile() as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
        try:
            for index in order:process.stdin.write(frames[index].tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('WebM encoding failed')
        except Exception as exc:
            process.kill();process.wait();log.seek(0)
            raise RuntimeError(log.read().decode(errors='replace')) from exc
    return command


def verify_webm(path,frames,order):
    """Decode every frame; check count and that both endpoints survived encoding."""
    executable=shutil.which('gst-launch-1.0')
    width,height=frames[0].size;frame_bytes=width*height*3
    with tempfile.TemporaryDirectory() as directory:
        decoded=Path(directory)/'decoded.rgb'
        subprocess.run([executable,'-q','filesrc',f'location={path}','!','matroskademux','!','vp8dec','!',
                        'videoconvert','!','video/x-raw,format=RGB','!','filesink',f'location={decoded}'],check=True,capture_output=True)
        if decoded.stat().st_size!=len(order)*frame_bytes:raise AssertionError('decoded frame count differs')
        stats=[]
        with decoded.open('rb') as file:
            for index in (0,max(order)):
                position=order.index(index);file.seek(position*frame_bytes)
                actual=np.frombuffer(file.read(frame_bytes),np.uint8).reshape(height,width,3)
                error=np.abs(actual.astype(np.int16)-np.asarray(frames[index]).astype(np.int16))
                if float(error.mean())>3:raise AssertionError(f'video endpoint differs excessively from source: {error.mean():.3f}')
                stats.append(dict(frame=position,mean_abs_uint8=float(error.mean()),max_abs_uint8=int(error.max())))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=24)
    args=parser.parse_args()
    if args.fps<2:parser.error('fps must be at least 2')
    report=json.loads(args.report.read_text());size=report['size']
    row=next(row for row in report['results'] if row['name']=='Elastic')
    if row['parameters']!='Elastic(sigma=32, magnitude=8, border=True)':raise ValueError('expected the strong Elastic benchmark')
    case=next(case for case in catalog(size,np.zeros((1,size,size,size),np.float32)) if case.name=='Elastic')
    seed=row['preview_selection']['seed'];indices=row['preview_selection']['augmented_indices']
    with ZarrReader.open(report['array'],threads=1) as reader:
        image=reader.read(report['start'],(size,)*3).astype(np.float32)[None]/np.float32(255)
    sample=inputs(image)['both'];pipeline=A.Pipeline(case.transforms,backend='cpu')
    plan=pipeline.sample(sample,seed=seed);endpoint=plan.apply(sample)['image'][0]
    np.testing.assert_array_equal(endpoint,pipeline(sample,seed=seed)['image'][0])
    field=displacement(plan);rms=float(np.sqrt(np.mean(np.sum(field*field,axis=-1))))
    original=[np.take(image[0],i,axis=a) for a,i in enumerate(indices)]
    final=[np.take(endpoint,i,axis=a) for a,i in enumerate(indices)]
    samplers=slice_samplers(image[0],field,indices)
    for expected,actual in zip(original,sample_planes(image[0],samplers,0)):np.testing.assert_array_equal(actual,expected)
    for expected,actual in zip(final,sample_planes(image[0],samplers,1)):np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)
    fractions=(.5-.5*np.cos(np.linspace(0,np.pi,2*args.fps+1))).tolist();fractions[0]=0.;fractions[-1]=1.
    frames=[animation_frame(original,original if fraction==0 else final if fraction==1 else sample_planes(image[0],samplers,fraction),
                            indices,fraction,rms,seed) for fraction in fractions]
    last=len(frames)-1
    order=[0]*args.fps+list(range(1,last))+[last]*args.fps+list(range(last-1,0,-1))
    output=args.report.parent/'animations';output.mkdir(parents=True,exist_ok=True)
    video=output/'Elastic.webm';poster=output/'Elastic-poster.webp';manifest=output/'Elastic-animation.json'
    frames[-1].save(poster,lossless=True,method=6)
    command=encode_webm(video,frames,order,args.fps)
    decoded=verify_webm(video,frames,order)
    metadata=dict(parameters=row['parameters'],seed=seed,indices=indices,achieved_rms=rms,fps=args.fps,
                  frames=len(order),duration_seconds=len(order)/args.fps,fractions=fractions,frame_order=order,
                  input_sha256=hashlib.sha256(image.tobytes()).hexdigest(),endpoint_sha256=hashlib.sha256(endpoint.tobytes()).hexdigest(),
                  field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),encoder_command=command,decoded_endpoint_checks=decoded,
                  interpolation='trilinear; nearest border; sampled production field scaled linearly; fixed slices and window [0,1]')
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    row['animation']={key:os.path.relpath(path,args.report.parent) for key,path in [('video',video),('poster',poster),('manifest',manifest)]}
    row['animation']['caption']=f'Strong cubic B-spline deformation: original → deformed → original. Fixed Z/Y/X slices, seed {seed}, achieved RMS {rms:.2f} voxels.'
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(video=str(video),bytes=video.stat().st_size,**{k:metadata[k] for k in ('achieved_rms','frames','duration_seconds','decoded_endpoint_checks')}),indent=2))


if __name__=='__main__':main()
