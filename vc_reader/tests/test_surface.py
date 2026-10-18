"""Analytic surface geometry, exact projection and CPU/CUDA parity."""
import os
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

if os.environ.get('VC_READER_NATIVE_DIR'):
    from bench_pipeline import load_extensions
    load_extensions(Path(os.environ['VC_READER_NATIVE_DIR']))
from vc_reader import Affine, Source, Storage, Surface, SurfaceRegistration, VolumeRegistration, Companion
from vc_reader.sampling import open_sample
from vc_reader.cuda import is_available
from test_source_storage import array


def host(a):
    return a.numpy() if hasattr(a,'numpy') else a


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        z,y,x=np.indices((12,14,16),dtype=np.float32)
        self.image=z*100+y*10+x
        self.reference=self.source('image',self.image,('z','y','x'),role='image',
            registration=VolumeRegistration('s','v',affine=Affine.identity(),volume_spacing_um=(1,1,1)))
        y,x=np.indices((8,9),dtype=np.float32)
        self.xyz=np.stack((x+3,y+2,np.full_like(x,5)))
        coords=self.source('xyz',self.xyz,('c','y','x'),role='scalar')
        self.surface=Surface('surface',coords.storage,'s','v',axes=('c','y','x'),component_order=('x','y','z'),
            xyz_to_volume=Affine.identity(),pixel_to_uv=((1,0,0),(0,1,0)),spacing_um=(1,1,1))
        self.labels=(y*9+x).astype(np.uint64)+(1<<40)
        self.label=self.source('ink',self.labels,('y','x'),role='label',registration=SurfaceRegistration(self.surface,((1,0,0),(0,1,0))))

    def source(self,name,data,axes,**kwargs):
        path=self.root/(name+'.zarr');array(path,data,{'_ARRAY_DIMENSIONS':axes})
        return Source(name,Storage(str(path),'zarr'),**kwargs)

    def backends(self):
        return ('cpu','cuda') if is_available() else ('cpu',)

    def test_layers_and_categorical_exactness(self):
        for backend in self.backends():
            with self.subTest(backend=backend),open_sample(self.reference,{'ink':self.label},backend=backend) as reader:
                result=reader.render_surface(self.surface,(1,2),(3,4),offsets_um=(-1,0,2))
                expected=np.stack([self.image[z,3:6,5:9] for z in (4,5,7)])[None]
                np.testing.assert_allclose(host(result.data['image']),expected)
                np.testing.assert_array_equal(host(result.data['ink']),self.labels[None,1:4,2:6])
                self.assertTrue(host(result.valid['image']).all())
                with self.assertRaisesRegex(ValueError,'explicit'):
                    reader.read((0,0,0),(1,1,1))

    def test_projection_crop_padding_thickness_and_tiling(self):
        results=[]
        for backend in self.backends():
            with self.subTest(backend=backend),open_sample(self.reference,{'ink':self.label},backend=backend) as reader:
                result=reader.project_surface(self.surface,(2,2),(2,3),label_target='ink',padding_um=1)
                meta=result.metadata['projection'];self.assertEqual(meta['origin_zyx'],(4,2,3));self.assertEqual(meta['shape_zyx'],(3,6,7))
                z,y,x=meta['origin_zyx'];d,h,w=meta['shape_zyx']
                np.testing.assert_array_equal(host(result.data['image']),self.image[None,z:z+d,y:y+h,x:x+w])
                coverage=host(result.valid['ink']);self.assertEqual(int(coverage.sum()),15)
                np.testing.assert_array_equal(host(result.data['ink'])[coverage],self.labels[np.ix_([2,2,3],[2,2,3,4,4])].ravel())
                self.assertTrue(host(result.masks['ink']['conflict']).any())
                tiled=np.zeros_like(host(result.data['ink']));tilevalid=np.zeros_like(coverage)
                for tile in reader.iter_project_surface(self.surface,(2,2),(2,3),label_target='ink',padding_um=1,tile_size=(1,2,3)):
                    at=tuple(a-b for a,b in zip(tile.metadata['projection']['origin_zyx'],meta['origin_zyx']))
                    sl=(slice(None),*(slice(a,a+n) for a,n in zip(at,tile.data['ink'].shape[1:])))
                    tiled[sl]=host(tile.data['ink']);tilevalid[sl]=host(tile.valid['ink'])
                np.testing.assert_array_equal(tiled,host(result.data['ink']));np.testing.assert_array_equal(tilevalid,coverage)
                band=reader.project_surface(self.surface,(2,2),(2,3),label_target='ink',thickness_um=2)
                self.assertEqual(band.metadata['projection']['shape_zyx'][0],3)
                self.assertGreater(host(band.valid['ink']).sum(),15)
                results.append((host(result.data['ink']),coverage))
        for result in results[1:]:
            for actual,expected in zip(result,results[0]):np.testing.assert_array_equal(actual,expected)

    def test_validation_budget_worker_and_zero_labels(self):
        zero=replace(self.label,storage=self.source('zero',np.zeros((8,9),np.uint8),('y','x')).storage)
        with open_sample(self.reference,{'ink':zero}) as reader:
            result=reader.project_surface(self.surface,(2,2),(2,3),label_target='ink')
            self.assertEqual(int(result.valid['ink'].sum()),15);self.assertFalse(result.data['ink'].any())
            restored=pickle.loads(pickle.dumps(reader))
            with restored:
                np.testing.assert_array_equal(restored.project_surface(self.surface,(2,2),(2,3),label_target='ink').valid['ink'],result.valid['ink'])
            for option in ({'thickness_um':-1},{'padding_um':float('nan')}):
                with self.assertRaises(ValueError):reader.project_surface(self.surface,(2,2),(2,3),label_target='ink',**option)
        with open_sample(self.reference,{'ink':self.label},surface_output_bytes=1) as reader:
            with self.assertRaisesRegex(MemoryError,'iter_project_surface'):
                reader.project_surface(self.surface,(2,2),(2,3),label_target='ink')
        with self.assertRaisesRegex(ValueError,'pixel_to_uv'):
            open_sample(self.reference,{'ink':replace(self.label,registration=SurfaceRegistration(self.surface))})

    def test_tifxyz_xy_scale_holes_boundaries_and_normal_sign(self):
        import json,tifffile
        root=self.root/'surface';root.mkdir()
        for axis,values in zip('xyz',self.xyz):tifffile.imwrite(root/(axis+'.tif'),values)
        (root/'meta.json').write_text(json.dumps({'scale':[.5,.25]}))
        mask=np.ones((8,9),np.uint8);mask[3,3]=0;tifffile.imwrite(root/'mask.tif',mask)
        surf=replace(self.surface,geometry=Storage(str(root),'tifxyz'),axes=('y','x'),pixel_to_uv=None)
        for backend in self.backends():
            with open_sample(self.reference,backend=backend) as reader:
                result=reader.render_surface(surf,(4,2),(1,1),offsets_um=(1,))
                np.testing.assert_allclose(host(result.data['image']),self.image[None,6:7,3:4,4:5])
                hole=reader.render_surface(surf,(12,6),(1,1))
                self.assertFalse(host(hole.valid['image']).any())
                outside=reader.render_surface(surf,(29,17),(1,1))
                self.assertFalse(host(outside.valid['image']).any())
        reverse=replace(self.surface,pixel_to_uv=((-1,0,8),(0,1,0)))
        with open_sample(self.reference) as reader:
            result=reader.render_surface(reverse,(1,6),(1,1),offsets_um=(1,))
            np.testing.assert_allclose(result.data['image'],self.image[None,4:5,3:4,5:6])

    def test_anisotropic_offsets_cross_volume_and_masks(self):
        from vc_reader import QueryContext,Frame
        mask=self.source('supervision',np.zeros((8,9),np.uint8),('y','x'),role='label')
        label=replace(self.label,companions=(Companion('supervision',mask,same_grid=True),))
        surf=replace(self.surface,spacing_um=(1,2,4))
        label=replace(label,registration=SurfaceRegistration(surf,((1,0,0),(0,1,0))))
        for backend in self.backends():
            with open_sample(self.reference,{'ink':label},backend=backend) as reader:
                layers=reader.render_surface(surf,(1,1),(1,1),offsets_um=(4,))
                np.testing.assert_allclose(host(layers.data['image']),self.image[None,6:7,3:4,4:5])
                result=reader.project_surface(surf,(2,2),(2,3),label_target='ink')
                self.assertTrue(host(result.valid['ink']).any())
                self.assertFalse(host(result.masks['ink']['supervision']).any())
        moved=replace(self.label,query=QueryContext(Frame('s','other'),'transform',Affine.from_rows(((1,0,0,1),(0,1,0,2),(0,0,1,1)))))
        ref=replace(self.reference,registration=replace(self.reference.registration,volume_id='other'))
        with open_sample(ref,{'ink':moved}) as reader:
            result=reader.project_surface(self.surface,(2,2),(2,3),label_target='ink')
            self.assertEqual(result.metadata['projection']['origin_zyx'],(6,5,5))

    def test_rotated_patch_and_folded_conflicts(self):
        rotated=replace(self.label,registration=SurfaceRegistration(self.surface,((.8,-.3,2),(.3,.8,1))))
        results=[]
        for backend in self.backends():
            with open_sample(self.reference,{'ink':rotated},backend=backend) as reader:
                result=reader.project_surface(self.surface,(1,1),(3,3),label_target='ink',thickness_um=.8,padding_um=1)
                shape=result.data['ink'].shape
                fields={key:host(value) for key,value in {'value':result.data['ink'],'coverage':result.valid['ink'],
                                                        'conflict':result.masks['ink']['conflict']}.items()}
                tiled={key:np.zeros_like(value) for key,value in fields.items()}
                for tile in reader.iter_project_surface(self.surface,(1,1),(3,3),label_target='ink',thickness_um=.8,padding_um=1,tile_size=(2,3,4)):
                    at=tuple(a-b for a,b in zip(tile.metadata['projection']['origin_zyx'],result.metadata['projection']['origin_zyx']))
                    sl=(slice(None),*(slice(a,a+n) for a,n in zip(at,tile.data['ink'].shape[1:])))
                    for key,array in {'value':tile.data['ink'],'coverage':tile.valid['ink'],'conflict':tile.masks['ink']['conflict']}.items():tiled[key][sl]=host(array)
                for key in fields:np.testing.assert_array_equal(tiled[key],fields[key])
                results.append(fields)
        for result in results[1:]:
            for key in result:np.testing.assert_array_equal(result[key],results[0][key])

    def test_invalid_geometry_and_unsupported_annotation(self):
        invalid=self.xyz.copy();invalid[:]=np.nan
        bad=replace(self.surface,geometry=self.source('invalid',invalid,('c','y','x'),role='scalar').storage)
        label=replace(self.label,registration=SurfaceRegistration(bad,((1,0,0),(0,1,0))))
        with open_sample(self.reference,{'ink':label}) as reader:
            with self.assertRaisesRegex(ValueError,'no valid nondegenerate'):
                reader.project_surface(bad,(1,1),(2,2),label_target='ink')
        other=replace(self.label,id='other',registration=SurfaceRegistration(self.surface,((1,0,1000),(0,1,1000))))
        with open_sample(self.reference,{'ink':self.label,'other':other}) as reader:
            result=reader.project_surface(self.surface,(1,1),(2,2),label_target='ink')
            self.assertFalse(result.valid['other'].any());self.assertTrue(result.valid['ink'].any())

    def test_folded_overlap_exact_values_and_large_dense_output(self):
        xyz=self.xyz.copy();xyz[0]=np.abs(xyz[0]-7)+3
        surf=replace(self.surface,geometry=self.source('fold',xyz,('c','y','x'),role='scalar').storage)
        label=replace(self.label,registration=SurfaceRegistration(surf,((1,0,0),(0,1,0))))
        outputs=[]
        for backend in self.backends():
            with open_sample(self.reference,{'ink':label},backend=backend) as reader:
                result=reader.project_surface(surf,(1,0),(3,9),label_target='ink')
                self.assertTrue(host(result.masks['ink']['conflict']).any())
                outputs.append(host(result.data['ink']))
        for output in outputs[1:]:np.testing.assert_array_equal(output,outputs[0])
        big=replace(self.surface,xyz_to_volume=Affine.from_rows(((12,0,0,0),(0,1,0,0),(0,0,1,0))))
        label=replace(self.label,registration=SurfaceRegistration(big,((1,0,0),(0,1,0))))
        outputs=[]
        for backend in self.backends():
            with open_sample(self.reference,{'ink':label},backend=backend) as reader:
                result=reader.project_surface(big,(2,1),(2,7),label_target='ink')
                self.assertGreater(result.data['ink'].shape[-1],64)
                self.assertFalse(host(result.valid['image']).any())
                outputs.append(host(result.data['ink']))
        for output in outputs[1:]:np.testing.assert_array_equal(output,outputs[0])

    def test_explicit_normal_and_geometry_mask_registration(self):
        from vc_reader import ValueFrame
        normals=np.zeros_like(self.xyz);normals[2]=-1
        normal=self.source('normal',normals,('c','y','x'),role='normal',
            registration=SurfaceRegistration(self.surface,((1,0,0),(0,1,0))),
            value_frame=ValueFrame(sample_id='s',volume_id='v',level=0),component_order=('x','y','z'))
        mask=np.ones((8,9),np.uint8);mask[3,3]=0
        mask=self.source('geometry-mask',mask,('y','x'),role='label',registration=SurfaceRegistration(self.surface,((1,0,0),(0,1,0))))
        surf=replace(self.surface,normals=normal,mask=mask)
        for backend in self.backends():
            with open_sample(self.reference,backend=backend) as reader:
                result=reader.render_surface(surf,(1,1),(1,1),offsets_um=(1,))
                np.testing.assert_allclose(host(result.data['image']),self.image[None,4:5,3:4,4:5])
                result=reader.render_surface(surf,(3,3),(1,1))
                self.assertFalse(host(result.valid['image']).any())

    def test_face_contact_and_physical_band_metric(self):
        xyz=self.xyz.copy();xyz[2]+=.5
        surf=replace(self.surface,geometry=self.source('face',xyz,('c','y','x'),role='scalar').storage)
        label=replace(self.label,registration=SurfaceRegistration(surf,((1,0,0),(0,1,0))))
        ref=replace(self.reference,registration=replace(self.reference.registration,volume_spacing_um=(1,1,4)))
        for backend in self.backends():
            with open_sample(ref,{'ink':label},backend=backend) as reader:
                zero=reader.project_surface(surf,(2,2),(2,3),label_target='ink')
                self.assertEqual(zero.metadata['projection']['origin_zyx'][0],5)
                self.assertEqual(zero.data['ink'].shape[1],2)
                thin=reader.project_surface(surf,(2,2),(2,3),label_target='ink',thickness_um=7.9)
                self.assertEqual(thin.data['ink'].shape[1],2)
                thick=reader.project_surface(surf,(2,2),(2,3),label_target='ink',thickness_um=8)
                self.assertEqual(thick.data['ink'].shape[1],4)

    def test_bounded_index_local_reads_and_retained_outputs(self):
        from unittest.mock import patch
        y,x=np.indices((70,160),dtype=np.float32)
        xyz=np.stack((x/20+3,y/20+2,np.full_like(x,5)))
        surf=replace(self.surface,geometry=self.source('wide-grid',xyz,('c','y','x'),role='scalar').storage)
        labels=self.source('wide-labels',np.ones((70,160),np.uint8),('y','x'),role='label',registration=SurfaceRegistration(surf,((1,0,0),(0,1,0))))
        with open_sample(self.reference,{'ink':labels},plan_cache_bytes=64<<10,sampling_cache_bytes=2<<20) as reader:
            geom=next(iter(reader._geometries.values()))
            with patch.object(geom,'tile',wraps=geom.tile) as tiles:
                small=reader.project_surface(surf,(3,3),(4,4),label_target='ink')
                self.assertTrue(all(max(call.args[1])<=6 for call in tiles.call_args_list))
            result=reader.project_surface(surf,(0,0),(70,160),label_target='ink')
            saved=result.data['ink'].copy()
            with patch.object(geom,'tile',wraps=geom.tile) as tiles:
                reader.project_surface(surf,(0,0),(70,160),label_target='ink')
                self.assertEqual(tiles.call_count,0)
            stats=reader.stats()
            self.assertLessEqual(stats['plan_cache_bytes'],64<<10)
            self.assertLessEqual(stats['tile_cache_bytes'],2<<20)
        np.testing.assert_array_equal(result.data['ink'],saved)

    def test_big_endian_multichannel_labels_and_overlay_serialization(self):
        data=np.stack((self.labels%251,self.labels%193)).astype('>u2')
        label=self.source('colors',data,('c','y','x'),role='label',registration=self.label.registration)
        encoded=Source.from_dict(label.to_dict())
        self.assertEqual(encoded.registration.surface,self.surface)
        outputs=[]
        for backend in self.backends():
            with open_sample(self.reference,{'ink':encoded},backend=backend) as reader:
                layers=reader.render_surface(self.surface,(2,2),(2,3))
                np.testing.assert_array_equal(host(layers.data['ink']),data[:,2:4,2:5])
                result=reader.project_surface(self.surface,(2,2),(2,3),label_target='ink')
                if backend=='cpu':self.assertEqual(result.data['ink'].dtype,data.dtype)
                outputs.append(host(result.data['ink']))
        for output in outputs[1:]:np.testing.assert_array_equal(output,outputs[0])

    def test_large_frame_origins_preserve_fractional_geometry(self):
        huge=2**26
        translation=Affine.from_rows(((1,0,0,huge+.25),(0,1,0,huge+.25),(0,0,1,huge+.25)))
        reference=replace(self.reference,registration=replace(self.reference.registration,
            affine=Affine.from_rows(((1,0,0,huge),(0,1,0,huge),(0,0,1,huge)))))
        uv=((1,0,huge),(0,1,huge))
        surf=replace(self.surface,xyz_to_volume=translation,pixel_to_uv=uv)
        label=replace(self.label,registration=SurfaceRegistration(surf,uv))
        for backend in self.backends():
            with open_sample(reference,{'ink':label},backend=backend) as reader:
                layers=reader.render_surface(surf,(huge+2,huge+2),(2,3))
                expected=self.image[5,4:6,5:8]+np.float32(27.75)
                np.testing.assert_allclose(host(layers.data['image'])[0,0],expected,atol=1e-4)
                np.testing.assert_array_equal(host(layers.data['ink']),self.labels[None,2:4,2:5])
                projected=reader.project_surface(surf,(2,2),(2,3),label_target='ink')
                self.assertEqual(projected.metadata['projection']['origin_zyx'],(5,4,5))
                self.assertTrue(host(projected.valid['ink']).any())
