"""PBS-only axis-normal constraints on actual orientation transforms.

Independent of deferred0079/0080: those files/tests are not used or retried.
This suite names quarter-turn and transpose operations explicitly. Image-only
low-resolution simulation remains outside that coordinate-orientation contract.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.cli import build_parser
from vesuvius.models.utilities.cli_utils import update_config_from_args
import vesuvius.models.datasets.zarr_dataset as dataset_module
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from vesuvius.models.augmentation.transforms.utils.perf import collect_augmentation_names
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def objects(node):
    if node is None:return []
    children=[]
    for key in ['transforms','list_of_transforms']:
        value=getattr(node,key,None)
        if isinstance(value,(list,tuple)):children.extend(value)
    child=getattr(node,'transform',None)
    if child is not None:children.append(child)
    return [node]+[value for child in children for value in objects(child)]


def orientation(pipeline):
    return [n for n in objects(pipeline)if isinstance(n,(Rot90Transform,TransposeAxesTransform))]


def signature(pipeline):
    return sorted((type(n).__name__,tuple(sorted(n.allowed_axes)))for n in orientation(pipeline))


def expected(planes):
    return sorted((kind,tuple(plane))for kind in ['Rot90Transform','TransposeAxesTransform']for plane in planes)


def tree(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*')if p.is_file()}


def caller_pipeline(kind,mgr):
    name,clsname=('cross_frame_dataset.py','CrossFrameZarrDataset')if kind=='cross_frame'else('mutex_affinity_dataset.py','MutexAffinityDataset')
    path=Path(dataset_module.__file__).resolve().parent/name
    source=ast.parse(path.read_text(encoding='utf8'))
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name==clsname)
    namespace=dict(create_training_transforms=create_training_transforms,collect_augmentation_names=collect_augmentation_names)
    self=SimpleNamespace(mgr=mgr,is_training=True,patch_size=mgr.train_patch_size,
        _profile_augmentations=False,_get_skeleton_targets=lambda:([],{}),transforms=None)
    if kind=='mutex':
        method=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_initialize_transforms'))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        namespace[method.name](self)
    else:
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
        branch=deepcopy(next(n for n in init.body if isinstance(n,ast.If)and ast.unparse(n.test)=='self.is_training'))
        wrapper=ast.parse('def extracted(self,mgr,skeleton_targets,skeleton_ignore_values):\n    pass\n').body[0]
        wrapper.body=[branch];ast.fix_missing_locations(wrapper)
        exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(path),'exec'),namespace)
        namespace['extracted'](self,mgr,[],{})
    return self.transforms


class RotationAxesWiringTests(unittest.TestCase):
    def manager(self,*,axes=None,cli=None,shape=(8,8,8),stores=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        config={'tr_setup':{'model_name':'engineered_orientation','ckpt_out_base':str(root/'out')},
            'tr_config':{'patch_size':list(shape),'num_dataloader_workers':0},'model_config':{},
            'dataset_config':{'data_path':str(root),'targets':{'ink':{'out_channels':2}},
                'normalization_scheme':'none','skip_patch_validation':True,'no_spatial':False}}
        if axes is not None:config['dataset_config']['rotation_axes']=axes
        if stores:
            values=np.arange(np.prod(shape),dtype=np.uint16).reshape(shape)
            for path,data in [(root/'images/volume.zarr',values+100),(root/'labels/volume_ink.zarr',values)]:
                array=create_level_dataset(str(path),'0',shape,(4,)*len(shape),data.dtype,None);array[:]=data
        path=root/'config.yaml';path.write_text(json.dumps(config),encoding='utf8')
        mgr=ConfigManager(verbose=False);mgr.load_config(path)
        args=build_parser().parse_args(['--config',str(path),'--output',str(root/'out')]+(['--rotation-axes',cli]if cli is not None else []))
        update_config_from_args(mgr,args)
        return root,mgr,tree(root)

    def exercise(self,pipeline,shape,*,normal=None):
        grid=torch.stack(torch.meshgrid(*[torch.arange(n,dtype=torch.float32)for n in shape],indexing='ij'))
        encoded=(100*grid[0]+10*grid[1]+grid[2])[None]
        for operation in orientation(pipeline):
            for seed in [0,1,2,3]:
                np.random.seed(seed);torch.manual_seed(seed)
                out=operation(image=grid.clone(),segmentation=encoded.clone(),padding_mask=torch.ones((1,)+shape))
                self.assertEqual(tuple(out['image'].shape),(3,)+shape)
                self.assertTrue(torch.equal(out['segmentation'][0],100*out['image'][0]+10*out['image'][1]+out['image'][2]))
                self.assertTrue(torch.equal(out['padding_mask'],torch.ones((1,)+shape)))
                if normal is not None:self.assertTrue(torch.equal(out['image'][normal],grid[normal]))
                if isinstance(operation,Rot90Transform):self.assertFalse(torch.equal(out['image'],grid))

    def test_cli_z_keeps_depth_axis_in_zarr_pipeline(self):
        root,mgr,before=self.manager(cli='z',stores=True)
        self.assertEqual(tuple(mgr.allowed_rotation_axes),(0,))
        dataset=ZarrDataset(mgr,is_training=True)
        self.assertEqual(signature(dataset.transforms),expected([(1,2)]))
        self.exercise(dataset.transforms,(8,8,8),normal=0)
        self.assertEqual(tree(root),before)

    def test_other_single_axis_normals(self):
        observations=[];pipelines=[]
        for normal,plane in [(1,(0,2)),(2,(0,1))]:
            pipeline=create_training_transforms((8,8,8),allowed_rotation_axes=[normal])
            observations.append(signature(pipeline));pipelines.append((normal,pipeline))
        self.assertEqual(observations,[expected([(0,2)]),expected([(0,1)])])
        for normal,pipeline in pipelines:self.exercise(pipeline,(8,8,8),normal=normal)

    def test_two_normals_do_not_add_a_third_plane(self):
        pipeline=create_training_transforms((8,8,8),allowed_rotation_axes=[0,1])
        self.assertEqual(signature(pipeline),expected([(1,2),(0,2)]))
        self.exercise(pipeline,(8,8,8))

    def test_cli_none_disables_orientation_not_intensity(self):
        root,mgr,before=self.manager(cli='none',stores=True)
        self.assertEqual(tuple(mgr.allowed_rotation_axes),())
        dataset=ZarrDataset(mgr,is_training=True)
        self.assertEqual(signature(dataset.transforms),[])
        self.assertTrue(any(isinstance(n,SimulateLowResolutionTransform)for n in objects(dataset.transforms)))
        self.assertTrue(any(type(n).__module__.startswith('vesuvius.models.augmentation.transforms.intensity.')for n in objects(dataset.transforms)))
        self.assertEqual(tree(root),before)

    def test_anisotropic_equal_plane_admissibility(self):
        valid=create_training_transforms((4,8,8),allowed_rotation_axes=[0])
        invalid=create_training_transforms((4,8,8),allowed_rotation_axes=[1])
        self.assertEqual((signature(valid),signature(invalid)),(expected([(1,2)]),[]))
        self.exercise(valid,(4,8,8),normal=0)

    def test_cross_frame_and_mutex_forward_constraint(self):
        _,mgr,_=self.manager(axes=['z'])
        pipelines=[caller_pipeline(kind,mgr)for kind in ['cross_frame','mutex']]
        self.assertEqual([signature(p)for p in pipelines],[expected([(1,2)])]*2)
        for pipeline in pipelines:self.exercise(pipeline,(8,8,8),normal=0)

    def test_unrestricted_controls(self):
        default=create_training_transforms((8,8,8))
        explicit=create_training_transforms((8,8,8),allowed_rotation_axes=[0,1,2])
        old=expected([(0,1,2)])
        self.assertEqual((signature(default),signature(explicit)),(old,old))
        self.exercise(default,(8,8,8));self.exercise(explicit,(8,8,8))

    def test_no_spatial_factory_control(self):
        # Direct healthy factory control,not0079's failed caller/legacy suite.
        pipeline=create_training_transforms((8,8,8),no_spatial=True,allowed_rotation_axes=[0])
        self.assertEqual(signature(pipeline),[])
        self.assertTrue(any(isinstance(n,SimulateLowResolutionTransform)for n in objects(pipeline)))

    def test_2d_mirror_control(self):
        for axes in [None,[],[0]]:
            pipeline=create_training_transforms((8,8),allowed_rotation_axes=axes)
            self.assertEqual(signature(pipeline),[])
            self.assertEqual(sum(isinstance(n,MirrorTransform)for n in objects(pipeline)),1)


if __name__=='__main__':unittest.main(verbosity=2)
