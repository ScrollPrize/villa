"""`augment_on_device`: the training pipeline transforms a batch on the GPU with `Pipeline.batch`."""
import numpy as np
import pytest
import torch

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _batch(device, batch=2, size=48):
    rng = np.random.default_rng(5)
    image = torch.from_numpy(rng.normal(size=(batch, 1, size, size, size)).astype(np.float32)).to(device)
    label = torch.zeros(batch, 1, size, size, size, dtype=torch.float32, device=device)
    label[:, :, 12:36, 12:36, 12:36] = 1
    return {"image": image, "ink": label}


@cuda_only
def test_device_pipeline_transforms_a_batch_on_the_gpu():
    device = torch.device("cuda")
    pipeline = create_training_transforms(patch_size=(48, 48, 48), label_names=["ink"], backend="cuda")
    source = _batch(device)
    before = source["image"].clone()

    first = pipeline.batch(source, seeds=[11, 12], device=device)
    second = pipeline.batch(source, seeds=[11, 12], device=device)

    assert torch.equal(source["image"], before), "inputs are not modified"
    for key in ("image", "ink"):
        assert first[key].device.type == "cuda"
        assert first[key].shape == source[key].shape
        assert torch.isfinite(first[key]).all()
        assert torch.equal(first[key], second[key]), "the same seeds give the same batch"
    assert not torch.equal(first["image"][0], first["image"][1]), "each sample has its own random stream"
    assert not torch.equal(first["image"], before)
