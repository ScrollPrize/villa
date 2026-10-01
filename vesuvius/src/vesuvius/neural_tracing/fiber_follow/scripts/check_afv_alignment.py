"""Plot one reproducibly selected AFV fiber against its unregistered native CT.

Top row: CT alone. Bottom row: the same CT with only fiber segments lying
within the displayed thin slab. Includes three native planes and one local
fiber-aligned plane. Save raw voxels and coordinates alongside the PNGs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from vesuvius.neural_tracing.fiber_follow.regression.datasets import read_dataset_config
from vesuvius.neural_tracing.fiber_follow.shared.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.shared.remote_prefetch import RemotePrefetcher
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec,RemoteChunkedArray


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'datasets/automated_fiber_volumes/alignment'
    output.mkdir(parents=True,exist_ok=True)
    doc,digest=read_dataset_config(root/'configs/mixed_ct_datasets.json')
    rng=np.random.default_rng(7349)
    jobs=[]
    for source in doc['sources']:
        if source['kind']!='afv':continue
        fibers=AFVFibers(source['path'],source['grid_scale'],validation=source['validation'],
            split='validation',sha256=source['sha256'])
        fi=int(rng.integers(len(fibers)));fiber=fibers[fi]
        assert fibers.metadata['frame']['vc_open_data_coordinate_space']==source['coordinate_space']
        # Exactly the training mapping: AFV native XYZ / grid_scale, then multiply
        # by input_scale=grid_scale/ct_grid_scale to address CT voxels.
        input_scale=source['grid_scale']/source['ct_grid_scale']
        points=fiber.points*input_scale
        center=interp_at(fiber.points,fiber.s,[fiber.length/2])[0]*input_scale
        nearby=np.abs(fiber.s-fiber.length/2)*input_scale<=64
        _,_,basis=np.linalg.svd(points[nearby]-center,full_matrices=False)
        planes=[('XY',np.array([1.,0,0]),np.array([0.,1,0]),'X','Y'),
                ('XZ',np.array([1.,0,0]),np.array([0.,0,1]),'X','Z'),
                ('YZ',np.array([0.,1,0]),np.array([0.,0,1]),'Y','Z'),
                ('Fiber-aligned',basis[0],basis[1],'Along fiber','Across fiber')]
        radius=64
        coordinates=np.linspace(-radius,radius,257)
        a,b=np.meshgrid(coordinates,coordinates)
        samples=[]
        for _,u,v,_,_ in planes:
            normal=np.cross(u,v)
            world=center+a[...,None]*u+b[...,None]*v
            samples.append(np.stack([world+d*normal for d in (-1.,0.,1.)]))
        all_points=np.concatenate([s.reshape(-1,3) for s in samples])
        lo=np.floor(all_points.min(0)).astype(int)-2
        hi=np.ceil(all_points.max(0)).astype(int)+3
        spec=FiberVolumeSpec('',ct_zarr=source['ct'],ct_level=source['ct_level'],
            ct_grid_scale=source['ct_grid_scale'],grid_scale=source['grid_scale'],
            inputs='ct',load_presence=False,cache_dir=doc['cache_dir'])
        jobs.append((source,fibers.catalog[fi][0],fiber,points,center,planes,samples,lo,hi,spec))
    reports=[]
    with RemotePrefetcher(connections=8) as service:
        for source,fid,fiber,points,center,planes,samples,lo,hi,spec in jobs:
            print('Fetching',source['name'],'fiber',fid,'center XYZ',center.tolist(),flush=True)
            service.client.ensure_metadata(spec)
            ct=RemoteChunkedArray(spec.ct_zarr,spec.ct_level,spec.cache_dir,32<<20,cache_only=True)
            start,size=lo[::-1],(hi-lo)[::-1]
            service.client.ensure(ct,[(start,size)])
            raw=ct.read(start,size)
            np.save(output/f'{source["name"]}_ct.npy',raw)
            np.savez(output/f'{source["name"]}_fiber.npz',points_ct_xyz=points,
                points_training_xyz=fiber.points,center_ct_xyz=center,ct_origin_zyx=start)
            panels=[map_coordinates(raw.astype(np.float32),(s-lo)[...,::-1].reshape(-1,3).T,
                order=1,mode='constant',cval=np.nan).reshape(s.shape[:-1]).mean(0) for s in samples]
            assert np.isfinite(panels).all(), 'Display planes must fit within the downloaded CT block'
            low,high=np.nanpercentile(np.stack(panels),[1,99])
            fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
            for column,((title,u,v,xlabel,ylabel),panel) in enumerate(zip(planes,panels)):
                uv=np.stack(((points-center)@u,(points-center)@v),axis=1)
                normal=(points-center)@np.cross(u,v)
                visible=(np.abs(normal)<=1.5)&(np.abs(uv).max(1)<=radius)
                line=np.where(visible[:,None],uv,np.nan)
                for row in (0,1):
                    ax=axes[row,column]
                    ax.imshow(panel,cmap='gray',origin='lower',extent=(-radius-.25,radius+.25,-radius-.25,radius+.25),
                        vmin=low,vmax=high,interpolation='nearest')
                    if row:
                        ax.plot(line[:,0],line[:,1],color='#ff5d3d',lw=1.1,alpha=.9)
                        ax.plot(0,0,'+',color='#00e5ff',ms=9,mew=1.1)
                    ax.set_title(title+(' | fiber overlay' if row else ' | CT only'),fontsize=11)
                    ax.set_xlabel(xlabel+' offset (native voxels)')
                    ax.set_ylabel(ylabel+' offset (native voxels)')
                    ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius)
            fig.suptitle(f'{source["name"]} — fiber {fid} ({fiber.tag})\n'
                f'Native CT XYZ center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}); '
                f'{source["native_voxel_size_um"]} µm/voxel',fontsize=14)
            fig.supxlabel('3-voxel mean slabs. Orange = AFV centerline within ±1.5 voxels of the plane; '
                'cyan + = fiber midpoint. No fitted registration or coordinate offset.',fontsize=10)
            png=output/f'{source["name"]}_alignment.png'
            fig.savefig(png,dpi=160);plt.close(fig)
            report=dict(source=source['name'],ct_url=source['ct'],afv_path=source['path'],
                coordinate_space=source['coordinate_space'],fiber_id=fid,fiber_name=fiber.name,
                family=fiber.tag,selection='seed 7349, uniform held-out fiber, arclength midpoint',
                center_ct_xyz=center.tolist(),ct_origin_zyx=start.tolist(),ct_shape_zyx=list(raw.shape),
                ct_level=spec.ct_level,grid_scale=spec.grid_scale,ct_grid_scale=spec.ct_grid_scale,
                voxel_size_um=source['native_voxel_size_um'],display_window=[float(low),float(high)],
                png=str(png),coordinate_registration='none',dataset_config_sha256=digest)
            (output/f'{source["name"]}_alignment.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
            print('Saved',png,flush=True)
    (output/'manifest.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
