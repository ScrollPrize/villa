"""Unsigned orientation geometry, augmentation isolation and checkpoint migration.

Runnable without pytest: python -m unittest discover -s tests -p test_direction_inputs.py
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from vesuvius.neural_tracing.fiber_follow.shared.direction_fields import (
    decode_direction_bytes, local_direction_moments, direction_crops,
)
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.data import (
    image_crop, ObservationBuilder, IdentityObservationBuilder, augment_image_pair, DirectTracer,
)
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_layout
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    save_checkpoint, load_checkpoint, checkpoint_config, build_parser,
)
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams


def array_at(path, values):
    path.mkdir(parents=True, exist_ok=True)
    (path/'.zarray').write_text(json.dumps(dict(shape=list(values.shape),chunks=list(values.shape),
        dtype='|u1',fill_value=0,order='C',filters=None,compressor=None,zarr_format=2)))
    (path/'0.0.0').write_bytes(values.astype(np.uint8).tobytes())


def encoded(n):
    n=np.asarray(n,float);n=n/np.linalg.norm(n,axis=-1,keepdims=True)
    n=np.where(n[...,2:3]<0,-n,n)
    return np.rint(n[...,:2]*127+128).astype(np.uint8)


def volume(root, nx=None, ny=None):
    shape=(40,40,40)
    nx=np.full(shape,204,np.uint8) if nx is None else nx
    ny=np.full(shape,77,np.uint8) if ny is None else ny
    for name,value in [('presence',np.full(shape,190,np.uint8)),('nx',nx),('ny',ny)]:
        array_at(root/f'fields/test_{name}.ome.zarr/3',value)
    z,y,x=np.indices((80,80,80))
    array_at(root/'ct/0',(x+2*y+z).clip(0,255).astype(np.uint8))
    return FiberVolume(FiberVolumeSpec(str(root/'fields'),ct_zarr=str(root/'ct'),ct_level=0,
                                     ct_grid_scale=4.,inputs='ct+presence'),cache_bytes=1<<20)


def config(**kwargs):
    options=dict(fine=CropSpec(depth=16,width=9,behind=7,spacing=.5),channels=4,hidden=16,
                 heads=2,layers=1,decoder_layers=1,n_future=4,n_history=8,
                 memory_slots=2,memory_steps=2,memory_stride=1,feature_detail_tokens=4)
    options.update(kwargs)
    return DirectConfig(**options)


def item(cfg):
    # Current crop deliberately rolled; memory frames have independent rolls/headings.
    frame=frame_from_heading(np.array([.3,.4,.8660254]))
    angle=.67;c,s=np.cos(angle),np.sin(angle)
    frame=frame@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return dict(pos=np.array([20.,20.,20.]),frame=frame,hist_local=np.zeros((cfg.n_history,3)),
        hmask=np.zeros(cfg.n_history),seed_valid=True,seed_pos=np.array([19.,20.,19.]),
        seed_tangent=np.array([.8,0,.6]),seed_age=2.,
        memory_track=dict(pos=np.array([[19.,20.,19.],[20.,20.,19.5]]),
            frame=np.stack([frame_from_heading(np.array([1.,0.,0.])),frame_from_heading(np.array([0.,1.,0.]))])))


class DirectionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_decode_axes_quantization_missing_and_normalization(self):
        a=np.array([255,128,128,0,0,255],np.uint8)
        b=np.array([128,255,128,0,128,255],np.uint8)
        got=decode_direction_bytes(a,b)
        np.testing.assert_allclose(got[:3],np.eye(3),atol=1e-7)
        np.testing.assert_array_equal(got[3:5],0)
        np.testing.assert_allclose(got[5],[2**-.5,2**-.5,0],atol=1e-7)
        with self.assertRaises(ValueError):decode_direction_bytes(a.astype(float),b)

    def test_sign_invariance_and_frame_tensor_law(self):
        rng=np.random.default_rng(13)
        n=rng.normal(size=(19,3));n/=np.linalg.norm(n,axis=-1,keepdims=True)
        frame=frame_from_heading(np.array([.4,-.8,.3]))
        actual=local_direction_moments(n,frame)
        np.testing.assert_array_equal(actual,local_direction_moments(-n,frame))
        q=np.einsum('ni,nj->nij',n,n)
        expected=frame.T@q@frame
        packed=np.stack([expected[:,a,b] for a,b in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],axis=-1)
        np.testing.assert_allclose(actual,packed,atol=2e-7)
        self.assertTrue((actual[:,3:]<0).any())
        with self.assertRaises(ValueError):local_direction_moments(n,2*frame)

    def test_decode_before_interpolation_across_hemisphere_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            nx=np.full((40,40,40),255,np.uint8);nx[:,:,21:]=1
            vol=volume(Path(tmp),nx,np.full_like(nx,128))
            crop=CropSpec(depth=1,width=1,behind=0)
            out=direction_crops([dict(pos=np.array([20.5,20.,20.]),frame=np.eye(3))],vol,crop)
            # Both endpoints describe the X axis. Averaging encoded bytes first
            # would produce 128/128 and invent a Z axis here.
            np.testing.assert_allclose(out.numpy().ravel(),[1,0,0,0,0,0],atol=1e-7)

    def test_sampling_matches_independent_world_tensor_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            z,y,x=np.indices((40,40,40))
            field=np.stack((.1+x/30.,-.9+y/45.,.3+z/60.),axis=-1)
            raw=encoded(field);vol=volume(Path(tmp),raw[...,0],raw[...,1])
            crop=CropSpec(depth=7,width=5,behind=3,spacing=.5)
            frame=frame_from_heading(np.array([.3,-.5,.8]));pos=np.array([19.2,20.3,18.4])
            state=dict(pos=pos,frame=frame)
            actual=direction_crops([state],vol,crop).numpy()[0]
            n=decode_direction_bytes(raw[...,0],raw[...,1]);world=np.einsum('...i,...j->...ij',n,n)
            points=(pos+crop_local_grid(crop)@frame.T)[...,::-1].reshape(-1,3).T
            interp=np.empty((points.shape[1],3,3))
            for a in range(3):
                for b in range(3):interp[:,a,b]=map_coordinates(world[...,a,b],points,order=1,mode='grid-constant',cval=0,prefilter=False)
            expected=frame.T@interp@frame
            packed=np.stack([expected[:,a,b] for a,b in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]).reshape(actual.shape)
            np.testing.assert_allclose(actual,packed,atol=3e-7)
            self.assertGreaterEqual(np.linalg.eigvalsh(expected).min(),-1e-7)
            np.testing.assert_allclose(actual[:3].sum(0),1,atol=3e-7)
            with ThreadPoolExecutor(2) as pool:
                threaded=direction_crops([state,state],vol,crop,pool).numpy()
            np.testing.assert_array_equal(threaded,np.stack([actual,actual]))
            # CT is sampled at twice the source resolution; directions stay on presence's grid.
            with_fields=image_crop([state],vol,crop,directions=True)
            np.testing.assert_array_equal(with_fields[0,2:],actual)
            torch.testing.assert_close(with_fields[:,:2],image_crop([state],vol,crop),rtol=0,atol=0)

    def test_padding_is_zero_without_inventing_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            vol=volume(Path(tmp),np.full((40,40,40),255,np.uint8),np.full((40,40,40),128,np.uint8))
            crop=CropSpec(depth=1,width=1,behind=0)
            out=direction_crops([dict(pos=np.array([-.5,10.,10.]),frame=np.eye(3)),
                                 dict(pos=np.array([-3.,10.,10.]),frame=np.eye(3))],vol,crop).numpy()
            np.testing.assert_allclose(out[0].ravel(),[.5,0,0,0,0,0],atol=1e-7)
            np.testing.assert_array_equal(out[1],0)

    def test_exact_byte_lookup_and_direct_output_for_multiple_items(self):
        from vesuvius.neural_tracing.fiber_follow.shared.direction_fields import _decoded_directions
        nx,ny=np.indices((256,256),dtype=np.uint8)
        np.testing.assert_array_equal(_decoded_directions().reshape(256,256,3),decode_direction_bytes(nx,ny))
        with tempfile.TemporaryDirectory() as tmp:
            vol=volume(Path(tmp));crop=CropSpec(depth=7,width=5,behind=3,spacing=.5)
            items=[dict(pos=np.array([20.,20.,20.]),frame=frame_from_heading(n))
                   for n in (np.array([0.,0.,1.]),np.array([.4,.3,.8]))]
            single=torch.cat([image_crop([i],vol,crop,directions=True) for i in items])
            with ThreadPoolExecutor(2) as pool:batched=image_crop(items,vol,crop,pool,directions=True)
            torch.testing.assert_close(batched,single,rtol=0,atol=0)
            output=np.full((2,8,7,5,5),np.nan,np.float32)
            direction_crops(items,vol,crop,out=output[:,2:])
            np.testing.assert_array_equal(output[:,2:],batched[:,2:].numpy())
            self.assertTrue(np.isnan(output[:,:2]).all())
            with self.assertRaisesRegex(ValueError,'contiguous'):
                direction_crops(items,vol,crop,out=output[:,2:,:,:,:][:,:,:,:,::-1])

    def test_paths_use_exact_presence_siblings_and_fail_for_bad_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);vol=volume(root)
            self.assertIsNone(vol._directions)
            fields=vol.direction_fields()
            self.assertTrue(fields[0].path.endswith('test_nx.ome.zarr/3'))
            self.assertIs(fields,vol.direction_fields())
            meta=root/'fields/test_ny.ome.zarr/3/.zarray'
            d=json.loads(meta.read_text());d['shape'][0]-=1;meta.write_text(json.dumps(d))
            other=FiberVolume(vol.spec)
            with self.assertRaisesRegex(ValueError,'presence grid'):other.direction_fields()
            meta.unlink()
            with self.assertRaises(FileNotFoundError):FiberVolume(vol.spec).direction_fields()

    def test_matching_shapes_with_misaligned_physical_grids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);vol=volume(root)
            meta=dict(multiscales=[dict(axes=['z','y','x'],datasets=[dict(path='3',coordinateTransformations=[dict(type='scale',scale=[8,8,8])])])])
            for name in ('presence','nx','ny'):
                (root/f'fields/test_{name}.ome.zarr/.zattrs').write_text(json.dumps(meta))
            vol.direction_fields()
            meta['multiscales'][0]['datasets'][0]['coordinateTransformations'].append(dict(type='translation',translation=[1,0,0]))
            (root/'fields/test_ny.ome.zarr/.zattrs').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'scale/origin'):FiberVolume(vol.spec).direction_fields()

    def test_main_memory_and_seed_have_their_own_frames_and_trace_parity(self):
        cfg=config(direction_inputs=True)
        with tempfile.TemporaryDirectory() as tmp:
            vol=volume(Path(tmp));state=item(cfg);builder=ObservationBuilder(cfg)
            x=builder.images([state],vol)
            world=decode_direction_bytes(np.array([204],np.uint8),np.array([77],np.uint8))[0]
            observations,seed=memory_layout(state,cfg)
            expected=local_direction_moments(world,state['frame'])
            np.testing.assert_allclose(x['fine'][0,2:,:,4,4].numpy(),np.repeat(expected[:,None],cfg.fine.depth,1),atol=2e-7)
            seed_x=x['feature_seed_x']
            np.testing.assert_allclose(seed_x['fine'][0,2:,7,4,4],local_direction_moments(world,seed['frame']),atol=2e-7)
            self.assertFalse(np.allclose(x['fine'][0,2:,7,4,4],seed_x['fine'][0,2:,7,4,4]))
            # Actual DirectTracer uses the same sampling path; exercise reconstructed
            # causal history rather than the optional explicit memory_track.
            plain={k:v for k,v in state.items() if k!='memory_track'}
            expected_x=builder.images([plain],vol)
            model=DirectFollower(cfg)
            tracer=DirectTracer(model,vol,cfg.fine,cfg.n_history,TraceParams(n_commit=4),device='cpu')
            try:
                actual=tracer.build_inputs(np.array([plain['pos']]),np.array([plain['frame']]),
                    np.array([plain['hist_local']]),np.array([plain['hmask']]),[plain])
                for k in expected_x:torch.testing.assert_close(actual[k],expected_x[k],rtol=0,atol=0)
            finally:tracer.close()
            warm=dict(plain,memory_warm=True)
            online=builder.images([warm],vol)
            torch.testing.assert_close(online['fine'],expected_x['fine'])
            self.assertNotIn('feature_seed_x',online)

    def test_all_image_augmentations_leave_directions_bitwise_unchanged(self):
        cfg=config(direction_inputs=True)
        g=torch.Generator().manual_seed(13)
        patch_image=torch.rand(8,9,9,9,generator=g)*2-1
        pair=patch_image.clone();before=pair[2:].clone()
        augment_image_pair(pair,(1.3,.1,.07),np.random.default_rng(7),blur_sigma=1.,drop_presence=True)
        torch.testing.assert_close(pair[2:],before,rtol=0,atol=0)
        self.assertEqual(pair[1].count_nonzero().item(),0)
        original=dict(x=dict(fine=patch_image[None].clone(),seed_mask=torch.ones(1,1),
            memory_patches=patch_image[None,None].repeat(1,3,1,1,1,1),memory_mask=torch.tensor([[True,False,True]]),
            memory_seed_patch=patch_image[None].clone(),memory_seed_valid=torch.tensor([True])))
        training=copy.deepcopy(original);builder=IdentityObservationBuilder(cfg,augment=True)
        state=dict(photometric=(1.3,.1,.07),identity_seed=3,blur_sigma=1.,drop_presence=True)
        with patch.object(ObservationBuilder,'__call__',return_value=training), \
             patch.object(builder,'identity_targets',return_value=dict(presence_dropped=torch.zeros(1))):
            got=builder([state],None)['x']
        for key in ('fine','memory_seed_patch'):
            torch.testing.assert_close(got[key][:,2:],original['x'][key][:,2:],rtol=0,atol=0)
        torch.testing.assert_close(got['memory_patches'][:,:,2:],original['x']['memory_patches'][:,:,2:],rtol=0,atol=0)



if __name__=='__main__':unittest.main()
