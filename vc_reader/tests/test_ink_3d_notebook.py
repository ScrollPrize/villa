"""Dataset coverage, disk patches and surface blending in the 3D notebook."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Ink3DNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'examples/02_ink_training_3d.ipynb'
        notebook = json.loads(path.read_text())
        cls.notebook = notebook
        cls.helpers = dict(np=np, torch=torch, F=F)
        exec(''.join(notebook['cells'][18]['source']), cls.helpers)
        for index, name in ((6, 'supervised_tiles'), (13, 'PatchDataset')):
            tree = ast.parse(''.join(notebook['cells'][index]['source']))
            node = next(n for n in tree.body if getattr(n, 'name', None) == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
                 dict(cls.helpers, Dataset=Dataset), cls.helpers)

    def devices(self):
        return ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',)

    def test_supervision_grid_covers_negative_labels_and_partial_edges(self):
        support = np.ones((19, 22), dtype=bool)
        support[8:16, 8:16] = False
        target = np.zeros_like(support)  # An all-negative region still trains.
        records = self.helpers['supervised_tiles'](
            support, target, (45, 55), (16, 16), 2, origin=(8, 12))
        seen = np.zeros_like(support, dtype=np.uint8)
        self.assertEqual(len(records), 8)
        for record in records:
            y, x = record['start']
            h, w = record['size']
            self.assertLessEqual(y+h, 45)
            self.assertLessEqual(x+w, 55)
            sy, sx = (y-8)//2, (x-12)//2
            region = np.s_[sy:sy+(h+1)//2, sx:sx+(w+1)//2]
            seen[region] += support[region]
            self.assertEqual(record['scan_count'], int(support[region].sum()))
            self.assertEqual(record['scan_positive'], 0)
        np.testing.assert_array_equal(seen, support)
        self.assertEqual(records[-1]['size'], (5, 11))
        self.assertEqual(self.helpers['supervised_tiles'](
            np.zeros_like(support), target, (38, 44), (16, 16), 2), [])

    def test_disk_dataset_and_complete_training_epochs(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for index in range(5):
                shape = (1, 4+index, 5, 6)
                ct = np.full(shape, 20+index*40, np.uint8)
                sheet = np.ones(shape, bool)
                target = np.zeros(shape, bool)
                target[:, ::2] = True
                support = np.ones(shape, bool)
                support[..., 0] = False
                path = Path(directory) / f'{index}.npz'
                np.savez(path, ct=ct, sheet=sheet, target=target, support=support)
                records.append(dict(path=path, count=int(support.sum()),
                                    positive=int((target & support).sum())))
            dataset = self.helpers['PatchDataset'](records)
            sample = dataset[2]
            self.assertEqual(sample['image'].shape, (2, 6, 5, 6))
            self.assertEqual(sample['image'].dtype, torch.float32)
            self.assertEqual(sample['target'].dtype, torch.float32)
            self.assertEqual(sample['support'].dtype, torch.bool)
            torch.testing.assert_close(sample['image'][0], torch.full((6, 5, 6), 100/255))
            self.assertTrue((sample['image'][1] == 1).all())
            train = self.helpers['PatchDataset'](records[:3])
            validation = self.helpers['PatchDataset'](records[3:])
            for device in self.devices():
                with self.subTest(device=device):
                    model = torch.nn.Conv3d(2, 1, 1).to(device)
                    namespace = dict(torch=torch, nn=torch.nn, F=F, DEVICE=torch.device(device),
                                     DataLoader=DataLoader, model=model, train_data=train,
                                     prepared={'train': records[:3]},
                                     train_loader=DataLoader(train, batch_size=1, shuffle=True),
                                     validation_loader=DataLoader(validation, batch_size=1))
                    with redirect_stdout(io.StringIO()):
                        exec(''.join(self.notebook['cells'][16]['source']), namespace)
                    self.assertEqual([h['patches'] for h in namespace['history']], [3, 3, 3])
                    self.assertTrue(all(np.isfinite(h['loss']) for h in namespace['history']))
                    expected = sum(r['positive'] for r in records[:3])
                    self.assertEqual(namespace['positive'], expected)

    def test_overlap_covers_non_aligned_end_and_preserves_constant_prediction(self):
        starts = self.helpers['overlap_starts'](23, 8, 5)
        self.assertEqual(starts, [0, 5, 10, 15])
        self.assertEqual(self.helpers['overlap_starts'](24, 8, 5), [0, 5, 10, 15, 16])
        for device in self.devices():
            with self.subTest(device=device):
                window = self.helpers['blend_window']((8, 8), device)
                self.assertTrue((window > 0).all().item())
                sums = torch.zeros(23, 24, device=device)
                weights = torch.zeros_like(sums)
                for y in starts:
                    for x in self.helpers['overlap_starts'](24, 8, 5):
                        sums[y:y+8, x:x+8] += window*.37
                        weights[y:y+8, x:x+8] += window
                self.assertTrue((weights > 0).all().item())
                torch.testing.assert_close(sums/weights, torch.full_like(sums, .37))
        with self.assertRaises(ValueError):
            self.helpers['overlap_starts'](24, 8, 9)

    def test_surface_plane_axis_order_fractional_coordinates_origins_and_invalidity(self):
        # The local coordinates stay precise even when the volume origin is large.
        origin = (100000001, 200000003, 300000007)  # ZYX
        delta = (1, -2, 3)  # XYZ
        base = tuple(o+d for o, d in zip(reversed(origin), delta))
        local = np.array([[[1.25, 2.5, 3.75], [5.5, 4.25, 2.5], [10., 8., 6.]],
                          [[-1., 2., 3.], [2., 3., 4.], [np.nan, 2., 3.]]], np.float32)
        geometry_valid = np.array([[True, True, True], [True, False, True]])
        points = local-np.asarray(delta, np.float32)
        for device in self.devices():
            with self.subTest(device=device):
                z, y, x = torch.meshgrid(torch.arange(7, device=device, dtype=torch.float32),
                                        torch.arange(9, device=device, dtype=torch.float32),
                                        torch.arange(11, device=device, dtype=torch.float32), indexing='ij')
                field = 10*z+2*y+x
                values = torch.stack((field, torch.ones_like(field)))
                sampled, valid = self.helpers['surface_plane'](values, points, geometry_valid, base, origin)
                np.testing.assert_array_equal(valid.cpu(), [[True, True, True], [False, False, False]])
                expected = torch.as_tensor(10*local[0, :, 2]+2*local[0, :, 1]+local[0, :, 0], device=device)
                torch.testing.assert_close(sampled[0, 0], expected)
                torch.testing.assert_close(sampled[1, 0], torch.ones(3, device=device))
                self.assertTrue(torch.isfinite(sampled).all().item())
