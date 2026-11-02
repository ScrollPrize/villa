"""Training and CLI contracts after moving image operations to vc_reader."""
import numpy as np
import pytest
import torch

from vesuvius.models.training.loss.losses import PlanarityLoss
from vesuvius.models.training.auxiliary_tasks.aux_structure_tensor import compute_structure_tensor
from vesuvius.models.training.auxiliary_tasks.aux_inplane_direction import compute_inplane_direction


def test_planarity_loss_backpropagates():
    image = torch.tensor(np.random.default_rng(3).normal(.5, .5, (1, 1, 9, 9, 9)), dtype=torch.float32, requires_grad=True)
    loss = PlanarityLoss(rho=.8)(image)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(13, 15), (11, 13, 15)])
@pytest.mark.parametrize("source", ["binary", "sdt"])
def test_auxiliary_components_and_direction(shape, source):
    mask = np.zeros(shape, np.uint8)
    mask[tuple(slice(3, -3) for _ in shape)] = 1
    tensor = compute_structure_tensor(mask, len(shape) == 2, compute_from=source)
    direction = compute_inplane_direction(mask, len(shape) == 2, compute_from=source)
    assert tensor.shape == ((3 if len(shape) == 2 else 6), *shape)
    assert direction.shape == (len(shape), *shape)
    assert np.isfinite(tensor).all() and np.isfinite(direction).all()
    np.testing.assert_array_equal(tensor[:, mask == 0], -100)
    np.testing.assert_array_equal(direction[:, mask == 0], -100)


def test_label_refinement_tiff_roundtrip(tmp_path):
    from vc_reader import images
    from vesuvius.label_processing.refine_labels import process_file
    image = np.zeros((17, 19), np.uint8)
    image[8, 3:16] = 1
    source = tmp_path / "source.tif"
    output = tmp_path / "out"
    output.mkdir()
    images.write_tiff(source, image)
    assert process_file(str(source), str(output)) == (str(source), True)
    result = images.read(output / source.name)
    assert result.shape == image.shape and result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 1}


def test_affine_json_singular_and_permuted_mesh(tmp_path):
    import json
    from vc_reader.affine import Affine
    from vc_reader.geometry import transform_points
    path = tmp_path / "transform.json"
    path.write_text(json.dumps({"transformation_matrix": [[2, 0, 0, 1], [0, 3, 0, 2], [0, 0, 0, 3]]}))
    with pytest.raises(ValueError):
        Affine.load_transform_json(str(path))
    transform = Affine.load_transform_json(str(path), allow_singular=True)
    assert transform.determinant == 0
    vertices = np.array([[1, 2, 3]], np.float32)
    perm = (2, 1, 0)
    result = transform_points(vertices[:, perm], np.asarray(transform.matrix, np.float32))[:, np.argsort(perm)]
    np.testing.assert_array_equal(result, [[3, 8, 7]])
