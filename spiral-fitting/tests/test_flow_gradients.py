from flow_fixtures import _make_small_spiral_model, _sample_scroll_points
import unittest
import pytest
import torch
import torch.nn.functional as F
import flow_triton
from flow_fields import (
    CartesianFlowField, CylindricalFlowField, BSplineFlowField,
    BSplineCylindricalFlowField, sample_field, sample_field_bspline,
)


def _reference_sampler(flow, low, high):
    # Build the reference from raw parameter clones, bypassing get_sampler's
    # detached leaves and pending-gradient accumulation.
    if isinstance(flow, CartesianFlowField):
        field = F.interpolate(
            low, size=tuple(high.shape[2:]), mode='trilinear')[0] + high[0]
        return lambda points: sample_field(points, field)
    if isinstance(flow, BSplineFlowField):
        return lambda points: (
            sample_field_bspline(low[0], points)
            + sample_field_bspline(high[0], points))

    def sample(points):
        values = []
        for parameter, num_phi, offsets in (
                (low, flow._lr_num_phi, flow._lr_offsets),
                (high, flow._hr_num_phi, flow._hr_offsets)):
            field = parameter[0]
            n0 = int(num_phi[0])
            pinned = torch.cat(
                [torch.zeros_like(field[:, :, :n0]), field[:, :, n0:]], dim=2)
            values.append(type(flow)._sample_lattice(pinned, num_phi, offsets, points))
        return values[0] + values[1]
    return sample


@pytest.mark.parametrize('field_class', [
    CartesianFlowField, CylindricalFlowField,
    BSplineFlowField, BSplineCylindricalFlowField,
], ids=['cartesian', 'cylindrical', 'bspline', 'bspline_cylindrical'])
def test_streamed_backwards_match_dense_autograd(field_class):
    cartesian = field_class is CartesianFlowField
    torch.manual_seed(4 if cartesian else 11)
    flow = field_class(torch.tensor([12, 12, 12]), spatial_scale_factor=6)
    with torch.no_grad():
        for parameter in flow.flows:
            parameter.normal_(std=0.1)
    batch_sizes = [37] if cartesian else [29, 41]
    points = [torch.rand(n, 3, requires_grad=True) for n in batch_sizes]
    reference_points = [p.detach().clone().requires_grad_(True) for p in points]
    reference_fields = [p.detach().clone().requires_grad_(True) for p in flow.flows]
    reference_sample = _reference_sampler(flow, *reference_fields)
    reference_outputs = [reference_sample(p) for p in reference_points]
    if cartesian:
        reference_outputs[0].square().sum().backward()
    else:
        (reference_outputs[0].square().mean() + reference_outputs[1].abs().mean()).backward()

    sampler = flow.get_sampler(0)
    # Independent backwards WITHOUT retain_graph through one cached sampler.
    if cartesian:
        chunks = [sampler(chunk) for chunk in points[0].split(17)]
        for output in chunks:
            output.square().sum().backward()
        outputs = [torch.cat(chunks)]
    else:
        outputs = [sampler(p) for p in points]
        outputs[0].square().mean().backward()
        outputs[1].abs().mean().backward()
    flow.apply_accumulated_field_grad()

    output_tolerance = dict(rtol=1e-5, atol=1e-6) if cartesian else {}
    grad_tolerance = dict(rtol=2e-4, atol=2e-5) if cartesian else {}
    for actual, expected in zip(outputs, reference_outputs):
        torch.testing.assert_close(actual, expected, **output_tolerance)
    for actual, expected in zip(
            points + list(flow.flows), reference_points + reference_fields):
        torch.testing.assert_close(actual.grad, expected.grad, **grad_tolerance)
    if not cartesian:
        assert flow._pending_field_graphs is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flow_triton._HAS_TRITON,
    reason='needs CUDA and triton')
@pytest.mark.parametrize('kind', ['bspline', 'bspline_cylindrical'])
def test_full_model_grads_match_eager_path(monkeypatch, kind):
    def run(disable_triton):
        model = _make_small_spiral_model(23, kind, device='cuda')
        points = _sample_scroll_points(41, 5).cuda()
        with monkeypatch.context() as environment:
            environment.setenv('FIT_SPIRAL_TRITON', '0' if disable_triton else '1')
            transform = model.get_slice_to_spiral_transform()
            loss = (transform(points)[..., 1:].norm(dim=-1)
                    / model.get_dr_per_winding()).mean()
            loss.backward()
        model.flow_field.apply_accumulated_field_grad()
        return loss.detach(), {name: p.grad for name, p in model.named_parameters()}

    eager_loss, eager_grads = run(disable_triton=True)
    triton_loss, triton_grads = run(disable_triton=False)
    torch.testing.assert_close(triton_loss, eager_loss, rtol=1e-4, atol=1e-7)
    for name, eager_grad in eager_grads.items():
        triton_grad = triton_grads[name]
        if eager_grad is None and triton_grad is None:
            continue
        torch.testing.assert_close(
            triton_grad, eager_grad, rtol=2e-3, atol=1e-5,
            msg=lambda base, name=name: f'{name}: {base}')


class SharedTransformLeafTests(unittest.TestCase):
    def _loss_families(self, transform, dr_per_winding, points_a, points_b):
        spiral_a = transform(points_a)
        family_a = (spiral_a[..., 1:].norm(dim=-1) / dr_per_winding).mean()
        spiral_b = transform(points_b)
        family_b = spiral_b.square().mean() * 1.e-4 + dr_per_winding * 0.01
        return family_a, family_b

    def _check_streamed_leaf_backwards_match_combined(self, flow_field_type):
        reference = _make_small_spiral_model(23, flow_field_type)
        streamed = _make_small_spiral_model(23, flow_field_type)
        streamed.load_state_dict(reference.state_dict())
        points_a = _sample_scroll_points(31, 5)
        points_b = _sample_scroll_points(17, 6)

        transform = reference.get_slice_to_spiral_transform()
        family_a, family_b = self._loss_families(
            transform, reference.get_dr_per_winding(), points_a, points_b)
        (family_a + family_b).backward()
        reference.flow_field.apply_accumulated_field_grad()

        shared_outputs = streamed.get_shared_transform_tensors()
        shared_leaves = tuple(
            output.detach().requires_grad_(True) for output in shared_outputs)
        leaf_transform = streamed.get_slice_to_spiral_transform(shared=shared_leaves)
        leaf_a, leaf_b = self._loss_families(
            leaf_transform, shared_leaves[0], points_a, points_b)
        torch.testing.assert_close(leaf_a, family_a)
        torch.testing.assert_close(leaf_b, family_b)
        # One backward per family, WITHOUT retain_graph: every path shared
        # between families ends at a detached leaf.
        leaf_a.backward()
        leaf_b.backward()
        streamed.flow_field.apply_accumulated_field_grad()
        pending = [
            (output, leaf.grad) for output, leaf in zip(shared_outputs, shared_leaves)
            if output.requires_grad and leaf.grad is not None
        ]
        self.assertTrue(pending)
        torch.autograd.backward(
            [output for output, _ in pending], [grad for _, grad in pending])

        reference_grads = {name: p.grad for name, p in reference.named_parameters()}
        for name, parameter in streamed.named_parameters():
            reference_grad = reference_grads[name]
            if parameter.grad is None and reference_grad is None:
                continue
            torch.testing.assert_close(
                parameter.grad, reference_grad, rtol=1e-4, atol=1e-6,
                msg=lambda base, name=name: f'{name}: {base}')

    def test_cartesian_streamed_leaf_backwards_match_combined(self):
        self._check_streamed_leaf_backwards_match_combined('cartesian')

    def test_cylindrical_streamed_leaf_backwards_match_combined(self):
        self._check_streamed_leaf_backwards_match_combined('cylindrical')

    def test_bspline_streamed_leaf_backwards_match_combined(self):
        self._check_streamed_leaf_backwards_match_combined('bspline')

    def test_bspline_cylindrical_streamed_leaf_backwards_match_combined(self):
        self._check_streamed_leaf_backwards_match_combined('bspline_cylindrical')
