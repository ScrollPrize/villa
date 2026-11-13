"""Exercise the notebook's full-canvas split with real local paired readers."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from vc_reader import Catalog, Companion, Source, Storage
from test_source_storage import array


NOTEBOOK = Path(__file__).resolve().parents[1] / 'examples/01_ink_training_25d.ipynb'


class InkNotebookTests(unittest.TestCase):
    def namespace(self, root, supervision):
        def source(name, values, axes, role):
            path = root / name
            array(path, values, {'_ARRAY_DIMENSIONS': axes})
            return Source(name, Storage(str(path), 'zarr'), role=role, axes=axes)

        shape = supervision.shape
        render = source('ct', np.full((9, *shape), 127, np.uint8), ('z', 'y', 'x'), 'image').replace(
                         artifact_kind='layers-zarr')
        ink = np.zeros(shape, np.uint8)
        ink[1::3, 2::4] = 255
        mask = source('supervision', supervision, ('y', 'x'), 'label')
        label = source('ink', ink, ('y', 'x'), 'label').replace(
                        companions=(Companion('supervision', mask, True),))
        scroll = Catalog({'samples': {'s': {'sample': {'id': 's'}}}}).samples['s']
        scroll = scroll.with_source(render, kind='image')
        scroll = scroll.with_source(label, kind='label', image=render.id, same_grid=True)
        label = scroll.labels[label.id]
        return dict(np=np, torch=torch, Path=Path, Dataset=Dataset, DataLoader=DataLoader,
                    CACHE=root / 'cache', label=label, image_asset=label.surface_volumes.zarr(),
                    LEVEL=0, PATCH_YX=(8, 8), LAYERS=slice(0, 9), BATCH_SIZE=2,
                    VALIDATION_FRACTION=.2, options={'threads': 1})

    def run_cell(self, index, namespace):
        cell = json.loads(NOTEBOOK.read_text())['cells'][index]
        with redirect_stdout(io.StringIO()):
            exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell-{index}', 'exec'), namespace)

    def test_complete_disjoint_coverage_including_negative_and_edge_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            supervision = np.full((19, 21), 255, np.uint8)
            supervision[:8, 8:16] = 0  # skip one entirely unknown tile
            namespace = self.namespace(Path(directory), supervision)
            self.run_cell(8, namespace)
            self.addCleanup(namespace['patch_cache'].cleanup)
            self.run_cell(11, namespace)
            self.assertEqual(namespace['split_y'], 16)
            self.assertEqual(len(namespace['records']), 8)
            seen = np.zeros(supervision.shape, np.uint8)
            for split in ('train', 'validation'):
                dataset = namespace['train_data' if split == 'train' else 'validation_data']
                for index, record in enumerate(dataset.records):
                    patch = dataset[index]
                    self.assertEqual(patch['image'].dtype, torch.float32)
                    self.assertEqual(patch['image'].shape, (1, 9, 8, 8))
                    self.assertEqual(patch['target'].shape, (1, 8, 8))
                    self.assertEqual(patch['support'].dtype, torch.bool)
                    y, x = record['start']
                    height, width = min(8, 19-y), min(8, 21-x)
                    mask = patch['support'][0].numpy()
                    self.assertEqual(int(mask.sum()), record['count'])
                    self.assertFalse(mask[height:, :].any())
                    self.assertFalse(mask[:, width:].any())
                    seen[y:y+height, x:x+width] += mask[:height, :width]
                    self.assertEqual(y >= namespace['split_y'], split == 'validation')
            np.testing.assert_array_equal(seen, supervision > 0)
            self.assertEqual(sum(r['count'] for r in namespace['records']), int((supervision > 0).sum()))

            # All-negative labels must retain precisely the same coverage and split.
            ink_path = Path(directory) / 'ink'
            array(ink_path, np.zeros(supervision.shape, np.uint8), {'_ARRAY_DIMENSIONS': ('y', 'x')})
            self.run_cell(11, namespace)
            self.assertEqual(sum(r['positive'] for r in namespace['records']), 0)
            self.assertEqual(sum(r['count'] for r in namespace['records']), int(seen.sum()))
