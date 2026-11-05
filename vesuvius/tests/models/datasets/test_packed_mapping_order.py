"""0073 prepared generated-store checks; execute in a frozen PBS job only.

The actual dataset constructor/item loader and complete trainer DataLoader method
run, with configuration metadata and a CPU trainer namespace supplied. No trainer
constructor, model, optimizer training, actual TIF packing or scan input is used.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset
import vesuvius.models.datasets.zarr_dataset as dataset_module
from vesuvius.models.datasets.zarr_dataset import ZarrDataset


def _tree(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def _loaders(dataset, manager):
    # Execute this actual complete method without importing unrelated model/build
    # dependencies or constructing BaseTrainer. Its globals are real NumPy/Torch.
    path=Path(dataset_module.__file__).resolve().parents[1]/'training/train.py'
    source=ast.parse(path.read_text(encoding='utf8'))
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='BaseTrainer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_configure_dataloaders')
    namespace=dict(globals())
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    trainer=SimpleNamespace(mgr=manager,is_distributed=False,device=torch.device('cpu'))
    return namespace['_configure_dataloaders'](trainer,dataset,None)


class PackedMappingOrderTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)

    def fixture(self, flags, *, enabled=True, invalid=False):
        self.flags=list(flags);n=len(flags);shape=(8*n,8,8)
        self.images=np.zeros(shape,dtype=np.uint16);self.labels=np.zeros(shape,dtype=np.uint8)
        records=[]
        for i,flag in enumerate(flags):
            self.images[8*i:8*i+8]=100+i*17
            self.labels[8*i:8*i+8]=1 if flag else 0
            records.append({'index':i,'filename':f'engineered{i}.tif','zarr_position':[8*i,0,0],
                'sample_shape':[8,8,8],'has_labels':flag,'labeled_ratio':float(flag)})
        for path,values in [(self.root/'images/volume.zarr',self.images),(self.root/'labels/volume_ink.zarr',self.labels)]:
            array=create_level_dataset(str(path),'0',values.shape,(4,4,4),values.dtype,None);array[:]=values
        mapping={} if invalid else {'version':'1.0','target_name':'ink','num_levels':1,
            'layout':{'sample_shape':[8,8,8],'gap_size':0},'samples':records,
            'splits':{'train':[],'val':[],'unassigned':list(range(n))}}
        (self.root/'samples_mapping.json').write_text(json.dumps(mapping),encoding='utf8')
        manager=SimpleNamespace(data_path=self.root,train_patch_size=(8,8,8),targets={'ink':{'valid_patch_value':1}},
            dataset_config={'valid_patch_value':1},allow_gap_extension=False,cache_valid_patches=False,
            skip_patch_validation=invalid,unlabeled_foreground_enabled=enabled,normalization_scheme='none',
            skip_intensity_sampling=True,ome_zarr_resolution=0,seed=0,verbose=False,tr_val_split=.8,
            train_batch_size=1,train_num_dataloader_workers=0)
        self.before=_tree(self.root)
        return ZarrDataset(manager,is_training=False),manager

    def assert_items(self, dataset, *, mapping=True):
        for i,patch in enumerate(dataset.valid_patches):
            item=dataset[i];z=patch.position[0];source=z//8
            np.testing.assert_array_equal(item['image'].numpy()[0],self.images[z:z+8].astype(np.float32))
            np.testing.assert_array_equal(item['ink'].numpy()[0],self.labels[z:z+8].astype(np.float32))
            self.assertEqual(bool(item['is_unlabeled']),not self.flags[source])
            self.assertEqual(patch.patch_size,(8,8,8))
            self.assertEqual(patch.volume_name,'volume')
            self.assertEqual(patch.safe_boundary,(z+8,8,8) if mapping else None)
        self.assertEqual(_tree(self.root),self.before)

    def assert_split(self, dataset, manager):
        train,val,train_indices,val_indices=_loaders(dataset,manager)
        self.assertTrue(all(not bool(batch['is_unlabeled'].any()) for batch in val))
        unlabeled={i for i,p in enumerate(dataset.valid_patches) if p.is_unlabeled_fg}
        self.assertTrue(unlabeled.issubset(train_indices))
        self.assertTrue(unlabeled.isdisjoint(val_indices))
        self.assertEqual(set(train_indices)|set(val_indices),set(range(len(dataset))))
        self.assertEqual(set(train_indices)&set(val_indices),set())
        self.assertEqual(sum(1 for _ in train),len(train_indices))
        self.assertEqual(_tree(self.root),self.before)

    def test_mixed_mapping_keeps_actual_validation_labeled(self):
        dataset,manager=self.fixture([False,False,True])
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg),(1,2))
        self.assert_items(dataset)
        self.assert_split(dataset,manager)

    def test_mapping_metadata_partition_is_independent_of_scanner_enable(self):
        dataset,manager=self.fixture([False,True],enabled=False)
        self.assertEqual([p.is_unlabeled_fg for p in dataset.valid_patches],[False,True])
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg),(1,1))
        self.assert_items(dataset);self.assert_split(dataset,manager)

    def test_stable_grouping_preserves_positions_values_and_boundaries(self):
        dataset,manager=self.fixture([False,True,False,True])
        self.assert_items(dataset)
        self.assertEqual([p.position[0] for p in dataset.valid_patches],[8,24,0,16])
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg),(2,2))
        self.assert_split(dataset,manager)

    def test_already_grouped_mapping_control(self):
        dataset,manager=self.fixture([True,True,False,False])
        self.assertEqual([p.position[0] for p in dataset.valid_patches],[0,8,16,24])
        self.assert_items(dataset);self.assert_split(dataset,manager)

    def test_all_labeled_mapping_control(self):
        dataset,manager=self.fixture([True,True])
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg),(2,0))
        self.assert_items(dataset);self.assert_split(dataset,manager)

    def test_all_unlabeled_mapping_control(self):
        dataset,manager=self.fixture([False,False])
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg),(0,2))
        self.assert_items(dataset);self.assert_split(dataset,manager)

    def test_invalid_mapping_explicit_skip_fallback_control(self):
        dataset,_=self.fixture([True,False],invalid=True)
        # Plain enumeration remains deliberately unvalidated;no change to its
        # count convention,per-item label detection or absence of safe boundaries.
        self.assertEqual((dataset.n_fg,dataset.n_unlabeled_fg,len(dataset)),(2,0,2))
        self.assertEqual([p.is_unlabeled_fg for p in dataset.valid_patches],[False,False])
        self.assert_items(dataset,mapping=False)


if __name__=='__main__': unittest.main(verbosity=2)
