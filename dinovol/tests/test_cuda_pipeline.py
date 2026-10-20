"""CUDA integration contracts: reader bytes, point geometry, pairing and batching."""
import copy
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from vc_reader import transforms as A
from torch.nn import functional as F

from dinovol_2.dataset.cuda_pipeline import (
    CropView, CudaBatchProcessor, CudaDataLoader, SpatialPlan, normalize_batch, point_pipeline, spatial_transforms,
)
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.dataset.volume_reader import open_volume
from dinovol_2.ops.collate import build_dino_ibot_collate_fn


def write_array(root, data, level=0, chunk=16):
    path = root / str(level)
    path.mkdir(parents=True, exist_ok=True)
    (path / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=data.shape,
        chunks=[chunk] * 3, dtype=data.dtype.str, compressor=None, fill_value=0,
        order='C', filters=None, dimension_separator='.')))
    for index in itertools.product(*(range((n + chunk - 1) // chunk) for n in data.shape)):
        start = [n * chunk for n in index]
        block = np.zeros((chunk,) * 3, dtype=data.dtype)
        source = data[tuple(slice(n, n + chunk) for n in start)]
        block[tuple(slice(n) for n in source.shape)] = source
        (path / '.'.join(map(str, index))).write_bytes(block.tobytes())


def collate(shape=(16, 16, 16)):
    return build_dino_ibot_collate_fn(dict(global_crop_size=shape, patch_size=(4, 4, 4),
        mask_ratio_min_max=(.1, .5), mask_sample_probability=.5, dtype=torch.float32))


class CudaPipelineTests(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    def test_normalization_before_anisotropic_resize(self):
        raw = np.random.default_rng(9).integers(0, 256, (9, 13, 17), dtype=np.uint8)
        lo, hi = np.percentile(raw, (1, 99))
        clipped = np.clip(raw.astype(np.float32), lo, hi).astype(np.float32)
        median = np.median(clipped)
        normalized = (clipped - median) / (1.4826 * np.median(np.abs(clipped - median)))
        expected = F.interpolate(torch.from_numpy(normalized)[None, None], size=(12, 8, 20),
                                 mode='trilinear', align_corners=False)
        actual = CudaBatchProcessor({})([CropView(torch.from_numpy(raw), (12, 8, 20))])
        self.assertTrue(actual.is_cuda)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=1e-5)
        for scheme in ('robust', 'minmax', 'zscore', 'none'):
            result = normalize_batch(torch.zeros((2, 1, 8, 8, 8), dtype=torch.uint8), scheme)
            self.assertTrue(result.is_cuda)
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result.count_nonzero().item(), 0)

    def test_source_dtypes(self):
        expected = torch.linspace(0, 1, 64).reshape(1, 1, 4, 4, 4)
        for dtype in (torch.uint8, torch.uint16, torch.int16, torch.float32, torch.float64):
            raw = torch.arange(64).reshape(1, 1, 4, 4, 4).to(dtype)
            result = normalize_batch(raw, 'minmax')
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result.cpu(), expected)

    def test_points_match_cuda_voxel_movements(self):
        for shape in ((8, 8, 8), (4, 8, 8), (4, 6, 8)):
            points = torch.cartesian_prod(*(torch.arange(n) for n in shape)).float()
            raw = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
            processor = CudaBatchProcessor({'normalization_scheme': 'none'})
            for _ in range(20):
                planned = SpatialPlan(shape)(image=CropView(raw, shape), keypoints=points)
                view = planned['image']
                view.augment = False  # test geometry independently of intensity changes
                image = processor([view])[0, 0].cpu()
                moved = planned['keypoints'].long()
                torch.testing.assert_close(image[moved[:, 0], moved[:, 1], moved[:, 2]], raw.flatten(), rtol=0, atol=0)

    def test_all_cube_orientations(self):
        shape = (4,) * 3
        points = torch.cartesian_prod(*(torch.arange(n) for n in shape)).float()
        raw = torch.arange(64, dtype=torch.float32).reshape(shape)
        processor = CudaBatchProcessor({'normalization_scheme': 'none'})
        seen = set()
        for order in itertools.permutations('zyx'):
            for flips in itertools.product((False, True), repeat=3):
                axes = ''.join(a for a, flip in zip('zyx', flips) if flip)
                operations = (('transpose', ''.join(order)),) + ((('flip', axes),) if axes else ())
                moved = point_pipeline(operations).map_points({'points': points}, shape=shape, seed=0)['points'].long()
                image = processor([CropView(raw, shape, spatial=operations)])[0, 0].cpu()
                torch.testing.assert_close(image[moved[:, 0], moved[:, 1], moved[:, 2]], raw.flatten(), rtol=0, atol=0)
                seen.add(tuple(image.flatten().tolist()))
        self.assertEqual(len(seen), 48)

    def test_intensity_preserves_labels(self):
        processor = CudaBatchProcessor({})
        operations = [A.RicianNoise(std=2), *processor.scanner(16), *processor.intensity(16),
                      A.FDA(reference='style', blend=.2, fraction=.1, on='image'),
                      A.PixelDistributionAdaptation(reference='style', blend=.2, on='image')]
        # Exercise every intensity effect, rather than relying on random selection.
        operations = [getattr(A, op._constructor.__name__)(*op._args, **{**op._kwargs, 'p': 1}) for op in operations]
        geometry = (('transpose', 'xyz'), ('flip', 'zy'))
        image = torch.rand((2, 1, 16, 16, 16), device='cuda') * 200 + 1
        labels = torch.randint(4, image.shape, device='cuda', dtype=torch.int32)
        original = labels.clone()
        pipeline = A.Pipeline([*operations, *spatial_transforms(geometry)], targets={
            'image': A.Image(valid=None), 'mask': A.Label(), 'style': A.ScalarField()}, backend='cuda')
        result = pipeline.batch({'image': image, 'mask': labels, 'style': image.flip(0)}, seeds=[7, 9])
        expected = labels.permute(0, 1, 4, 3, 2).flip((2, 3))
        torch.testing.assert_close(result['mask'], expected, rtol=0, atol=0)
        torch.testing.assert_close(labels, original, rtol=0, atol=0)
        self.assertEqual(result['mask'].dtype, labels.dtype)
        self.assertTrue(result['image'].is_cuda)
        self.assertTrue(torch.isfinite(result['image']).all())

    def test_paired_views_and_local_batching(self):
        for recipe in ('full', 'dino'):
            processor = CudaBatchProcessor({'augmentation': recipe}, collate())
            samples = []
            for _ in range(2):
                globals_, teachers = [], []
                for _ in range(2):
                    raw = torch.randint(256, (19, 19, 19), dtype=torch.uint8)
                    teacher = CropView(raw.clone(), (16, 16, 16))
                    student = SpatialPlan((16,) * 3)(image=CropView(raw, (16,) * 3), paired_spatial_image=teacher)['image']
                    teacher.augment = True
                    globals_.append(student)
                    teachers.append(teacher)
                locals_ = [SpatialPlan((8,) * 3)(image=CropView(
                    torch.randint(256, (11, 11, 11), dtype=torch.uint8), (8,) * 3))['image'] for _ in range(4)]
                samples.append(dict(global_views=globals_, local_views=locals_, gram_teacher_views=teachers))
            with patch.object(processor, 'batch', wraps=processor.batch) as batch:
                result = processor(samples)
            torch.testing.assert_close(result['collated_global_crops'], result['collated_gram_teacher_crops'], rtol=0, atol=0)
            local_pipeline = processor.pipelines[('intensity', (8, 8, 8))]
            calls = [call for call in batch.call_args_list if call.args[0] is local_pipeline]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].args[1].shape[0], 8)
            self.assertEqual(result['collated_local_crops'].shape, (8, 1, 8, 8, 8))

    def test_dataset_workers_labels_tokens_and_gram(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = np.random.default_rng(4).integers(1, 255, (48, 48, 48), dtype=np.uint8)
            write_array(root / 'volume', raw)
            write_array(root / 'volume', raw[::2, ::2, ::2], level=1)
            surface = np.zeros_like(raw)
            surface[::4] = 255
            write_array(root / 'surface', surface)
            with open_volume(root / 'volume', 0) as reader:
                np.testing.assert_array_equal(reader[5:31, 9:27, 11:43], raw[5:31, 9:27, 11:43])
            config = dict(datasets=[dict(volume_path=str(root / 'volume'), volume_scale=0,
                surface_path=str(root / 'surface'), pixel_size_um=8)], epoch_length=4, vol_trim_pct=1,
                source_sampling_size=[32]*3, global_crop_size=[16]*3, local_crop_size=[8]*3,
                global_crop_scale=[.25,.5], local_crop_scale=[.05,.1], num_local_crops=4,
                gram_teacher_crop_size=[24]*3, gram_teacher_no_augmentations=True,
                normalization_scheme='robust', augmentation='dino', read_ahead=2, occupancy_level=1,
                cross_view_token_patch_size=[4]*3,
                object_points=dict(enabled=True, use_metadata_meshes=False, points_per_sample=16,
                                   points_per_object=8, max_objects=4))
            for workers in (0, 2):
                dataset = SSLZarrDataset(copy.deepcopy(config), do_augmentations=True)
                loader = CudaDataLoader(dataset, batch_size=2, num_workers=workers,
                    persistent_workers=bool(workers), pin_memory=True, collate_fn=collate())
                try:
                    for result in loader:
                        self.assertTrue(result['collated_global_crops'].is_cuda)
                        self.assertEqual(result['collated_gram_teacher_crops'].shape, (4, 1, 24, 24, 24))
                        self.assertEqual(result['global_token_targets'].shape, (4, 64, 3))
                        self.assertTrue(torch.isfinite(result['global_token_targets']).all())
                        self.assertEqual(result['object_point_coordinates'].shape, (4, 16, 3))
                        self.assertTrue(torch.isfinite(result['collated_global_crops']).all())
                finally:
                    if workers:
                        loader._iterator._shutdown_workers()
                    dataset.close()
            config.update(num_local_crops=0, gram_teacher_crop_size=None, epoch_length=2)
            dataset = SSLZarrDataset(config)
            try:
                result = next(iter(CudaDataLoader(dataset, batch_size=2, collate_fn=collate())))
                self.assertTrue(result['collated_local_crops'].is_cuda)
                self.assertEqual(len(result['collated_local_crops']), 0)
            finally:
                dataset.close()

    def test_training_step_and_monitor(self):
        from dinovol_2.pretrain import DinoIBOTPretrainer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_array(root / 'volume', np.random.default_rng(5).integers(1, 255, (40,)*3, dtype=np.uint8))
            config = dict(device='cuda', batch_size=2, num_workers=0, max_iterations=5,
                output_dir=str(root / 'output'), use_amp=True, amp_dtype='bfloat16', compile=False,
                ibot_targets='cross_view', monitor_batch_size=2, freeze_last_layer_steps=0,
                warmup_steps=0, lr=1e-4, lr_schedule='constant',
                model=dict(model_type='v3', embed_dim=96, depth=2, num_heads=4,
                    global_crops_size=[16]*3, local_crops_size=[8]*3, patch_size=[4]*3,
                    dino_out_dim=128, ibot_out_dim=128, dino_head_hidden_dim=64,
                    dino_head_bottleneck_dim=32, ibot_head_hidden_dim=64, ibot_head_bottleneck_dim=32),
                dataset=dict(datasets=[dict(volume_path=str(root / 'volume'), volume_scale=0)],
                    epoch_length=4, vol_trim_pct=1, source_sampling_size=[24]*3,
                    global_crop_size=[16]*3, local_crop_size=[8]*3, num_local_crops=2,
                    global_crop_scale=[.4,.8], local_crop_scale=[.1,.2], augmentation='dino'))
            trainer = DinoIBOTPretrainer(config)
            loader = trainer.build_dataloader()
            try:
                batch = next(iter(loader))
                report = trainer.verify_train_step(batch)
                self.assertTrue(report['forward']['checks']['all_passed'], report['forward']['checks'])
                self.assertTrue(report['train_step']['checks']['all_passed'], report['train_step']['checks'])
                first = trainer.build_monitor_batch(seed=7)
                second = trainer.build_monitor_batch(seed=7)
                self.assertTrue(first['collated_global_crops'].is_cuda)
                torch.testing.assert_close(first['collated_global_crops'], second['collated_global_crops'], rtol=0, atol=0)
            finally:
                trainer._close_dataloader(loader)
                trainer._close_auxiliary_datasets()
                trainer._finish_wandb()

    def test_inference_preprocessing(self):
        from dinovol_2.eval.napari_visualizer import normalize_volume, pad_volume_to_patch_size
        raw = np.arange(5*7*9, dtype=np.float32).reshape(1, 5, 7, 9)
        image = normalize_volume(raw, normalization_scheme='minmax')
        padded, shape = pad_volume_to_patch_size(image, (4, 4, 4))
        self.assertTrue(padded.is_cuda)
        self.assertEqual(shape, (8, 8, 12))
        torch.testing.assert_close(padded[:, :5, :7, :9], image)
        self.assertEqual(float(padded[0, -1, -1, -1]), 1.)

    def test_cuda_required(self):
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            CudaBatchProcessor({}, device='cpu')


if __name__ == '__main__':
    unittest.main()
