"""Compression geometry, including its single-axis resampling path."""
import pickle

import numpy as np
import pytest

from vc_reader import augment as A


@pytest.mark.parametrize("shape,axis", [
    ((7, 9), 0), ((7, 9), 1),
    ((5, 7, 9), 0), ((5, 7, 9), 1), ((5, 7, 9), 2),
    ((1, 3, 2), 0), ((1, 3, 2), 1), ((1, 3, 2), 2),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threshold", [-1., .5])
@pytest.mark.parametrize("strength", [0., .25, .95])
def test_compression_coordinates_and_replay(shape, axis, dtype, threshold, strength):
    image = np.random.default_rng(71).normal(size=(2, *shape)).astype(dtype)
    labels = (np.arange(image.size, dtype=np.uint64) + np.uint64(2**63)).reshape(image.shape)
    sample = {"image": image, "mask": labels}
    transform = A.SheetCompression(axes="zyx"[-len(shape):][axis], strength=strength,
                                  intensity_sigma=0, smoothing=0, threshold=threshold)
    pipeline = A.Pipeline([transform])

    # Independently construct the deformation, then evaluate the full bilinear /
    # trilinear stencil in the package's defined corner order.
    reference = image[0].astype(np.float64)
    norm = (reference - reference.min()) / (reference.max() - reference.min())
    compression = (norm < threshold).astype(float) if threshold >= 0 else 1 - norm
    field = -np.cumsum(compression * strength, axis=axis) * (shape[axis] - 1) / shape[axis]
    q = np.indices(shape, dtype=np.float64)
    q[axis] = np.clip(q[axis] + field, 0, shape[axis] - 1)
    lo = np.floor(q).astype(np.int64)
    fraction = q - lo
    expected = np.zeros_like(image, dtype=np.float64)
    for corner in range(1 << len(shape)):
        weight = np.ones(shape)
        indices = []
        for a in range(len(shape)):
            bit = (corner >> a) & 1
            weight *= fraction[a] if bit else 1 - fraction[a]
            indices.append(np.minimum(lo[a] + bit, shape[a] - 1))
        expected += weight * image[(slice(None), *indices)].astype(np.float64)
    nearest = tuple(np.rint(q).astype(np.int64))
    expected_labels = labels[(slice(None), *nearest)]

    plan = pipeline.sample(sample, seed=123)
    for output in (pipeline(sample, seed=123), pipeline.apply(sample, plan),
                   pickle.loads(pickle.dumps(plan)).apply(sample),
                   pipeline({k: v.copy() for k, v in sample.items()}, seed=123, inplace=True)):
        assert output["image"].tobytes() == expected.astype(dtype).tobytes()
        assert output["mask"].tobytes() == expected_labels.tobytes()


def test_compression_interpolates_negative_zero_like_general_geometry():
    reference = np.arange(35., dtype=np.float64).reshape(1, 5, 7)
    image = np.full_like(reference, -0.)
    pipeline = A.Pipeline([A.SheetCompression(image="reference", axes="y",
                                             intensity_sigma=0, smoothing=0)],
                          targets={"reference": A.Image(), "image": A.Image()})
    output = pipeline({"reference": reference, "image": image}, seed=123)["image"]
    assert (output == 0).all()
    assert not np.signbit(output).any()
