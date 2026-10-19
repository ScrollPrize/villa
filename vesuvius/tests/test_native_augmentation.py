"""DataLoader integration tests; optional native extension and Torch required."""
import pickle
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vc_reader")
from vc_reader import transforms as A
from vesuvius.models.augmentation.native import (
    AugmentationIndex, AugmentationSampler, AugmentedDataset,
)


class Patches:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"image": torch.full((1, 5, 7), float(index)),
                "mask": torch.full((1, 5, 7), index, dtype=torch.int64)}


def collect(dataset, workers, epoch):
    sampler = AugmentationSampler(torch.utils.data.SequentialSampler(dataset))
    sampler.set_epoch(epoch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=workers)
    return torch.cat([batch["image"] for batch in loader])


def test_dataset_pickle_and_worker_scheduling():
    dataset = AugmentedDataset(Patches(), A.Pipeline([A.GaussianNoise(std=.25)]), seed=42)
    dataset = pickle.loads(pickle.dumps(dataset))
    expected = collect(dataset, 0, 3)
    actual = collect(dataset, 2, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(expected, collect(dataset, 0, 4))


def test_persistent_workers_receive_epoch_and_draw_identity():
    dataset = AugmentedDataset(Patches(), A.Pipeline([A.GaussianNoise(std=.25)]), seed=42)
    sampler = AugmentationSampler([1, 1, 2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler,
                                       num_workers=2, persistent_workers=True)
    try:
        sampler.set_epoch(0)
        first = [batch["image"] for batch in loader]
        sampler.set_epoch(1)
        second = [batch["image"] for batch in loader]
        assert not torch.equal(first[0], first[1])
        assert not torch.equal(first[0], second[0])
        sampler.set_epoch(0)
        replay = [batch["image"] for batch in loader]
        for a, b in zip(first, replay):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_sampler_forwards_epoch_to_distributed_sampler():
    sampler = torch.utils.data.DistributedSampler(Patches(), num_replicas=2, rank=0, seed=7)
    wrapped = AugmentationSampler(sampler, view=2)
    wrapped.set_epoch(4)
    assert sampler.epoch == 4
    assert len(wrapped) == 3
    assert all(token.epoch == 4 and token.view == 2 for token in wrapped)


def test_zarr_dataset_native_hook_without_io():
    # Exercise the actual __getitem__ path using in-memory arrays after discovery.
    # This test does not require opening or downloading a volume.
    module = pytest.importorskip("vesuvius.models.datasets.zarr_dataset")
    from types import SimpleNamespace
    dataset = module.ZarrDataset.__new__(module.ZarrDataset)
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    dataset._patches = [module.PatchInfo(0, "scroll", (0, 0), (5, 7))]
    dataset._volumes = [SimpleNamespace(image_array=image, label_arrays={"ink": image > 10}, volume_id="scroll")]
    dataset.is_2d = True
    dataset.normalizer = None
    dataset.target_names = ["ink"]
    dataset._native_augmentation = A.Pipeline([A.Flip(axes="x", p_axis=1)],
        targets={"image": A.Image(), "ink": A.Label(), "padding_mask": A.ValidMask()})
    dataset._native_augmentation_seed = 17
    dataset._profile_augmentations = True
    dataset._initialize_transforms()
    assert dataset.transforms is None
    result = dataset[AugmentationIndex(0, epoch=2)]
    np.testing.assert_array_equal(result["image"].numpy()[0], image[:, ::-1])
    np.testing.assert_array_equal(result["ink"].numpy()[0], (image > 10)[:, ::-1])
    assert "_aug_perf" in result


def test_dataset_can_reuse_fresh_reader_style_buffers():
    pipeline = A.Pipeline([A.BrightnessAdd(.5)])
    safe = AugmentedDataset(Patches(), pipeline, seed=42)
    inplace = AugmentedDataset(Patches(), pipeline, seed=42, inplace=True)
    inplace = pickle.loads(pickle.dumps(inplace))
    token = AugmentationIndex(2, epoch=3, draw=4)
    torch.testing.assert_close(inplace[token]["image"], safe[token]["image"], rtol=0, atol=0)


def test_native_profile_uses_training_metric_key():
    dataset = AugmentedDataset(Patches(), A.Pipeline([A.BrightnessAdd(.5)]), profile=True)
    result = dataset[0]
    assert set(result['_aug_perf']) == {'0'}
    assert result['_aug_perf']['0'] >= 0
    assert '_transform_perf' not in result
