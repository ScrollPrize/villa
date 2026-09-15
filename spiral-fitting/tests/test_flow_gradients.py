from flow_fixtures import _make_small_spiral_model, _sample_scroll_points
import unittest
import torch
import torch.nn.functional as F
from flow_fields import CartesianFlowField, CylindricalFlowField, sample_field


class CartesianFlowGradientTests(unittest.TestCase):
    def test_streamed_backwards_match_dense_autograd(self):
        torch.manual_seed(4)
        resolution = torch.tensor([12, 12, 12])
        flow = CartesianFlowField(resolution, spatial_scale_factor=6)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)

        points = torch.rand(37, 3, requires_grad=True)
        reference_points = points.detach().clone().requires_grad_(True)
        reference_lr = flow.flows[0].detach().clone().requires_grad_(True)
        reference_hr = flow.flows[1].detach().clone().requires_grad_(True)

        reference_lr_up = F.interpolate(
            reference_lr,
            size=tuple(reference_hr.shape[2:]),
            mode='trilinear',
        )[0]
        reference_field = reference_lr_up + reference_hr[0]
        reference_output = sample_field(reference_points, reference_field)
        reference_loss = reference_output.square().sum()
        reference_loss.backward()

        sampler = flow.get_sampler(0)
        outputs = [sampler(chunk) for chunk in points.split(17)]
        for output in outputs:
            output.square().sum().backward()
        flow.apply_accumulated_field_grad()

        torch.testing.assert_close(torch.cat(outputs), reference_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(points.grad, reference_points.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(flow.flows[0].grad, reference_lr.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(flow.flows[1].grad, reference_hr.grad, rtol=2e-4, atol=2e-5)


class CylindricalFlowGradientTests(unittest.TestCase):
    def test_streamed_backwards_and_pending_field_grad_match_dense_autograd(self):
        torch.manual_seed(11)
        flow = CylindricalFlowField(torch.tensor([12, 12, 12]), spatial_scale_factor=6)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)

        points_a = torch.rand(29, 3, requires_grad=True)
        points_b = torch.rand(41, 3, requires_grad=True)
        reference_a = points_a.detach().clone().requires_grad_(True)
        reference_b = points_b.detach().clone().requires_grad_(True)
        reference_lr = flow.flows[0].detach().clone().requires_grad_(True)
        reference_hr = flow.flows[1].detach().clone().requires_grad_(True)

        n0_lr = int(flow._lr_num_phi[0])
        n0_hr = int(flow._hr_num_phi[0])
        reference_lr_field = torch.cat(
            [torch.zeros_like(reference_lr[0][:, :, :n0_lr]), reference_lr[0][:, :, n0_lr:]],
            dim=2)
        reference_hr_field = torch.cat(
            [torch.zeros_like(reference_hr[0][:, :, :n0_hr]), reference_hr[0][:, :, n0_hr:]],
            dim=2)

        def reference_sample(pts):
            return (
                CylindricalFlowField._sample_lattice(reference_lr_field, flow._lr_num_phi, flow._lr_offsets, pts)
                + CylindricalFlowField._sample_lattice(reference_hr_field, flow._hr_num_phi, flow._hr_offsets, pts)
            )

        reference_out_a = reference_sample(reference_a)
        reference_out_b = reference_sample(reference_b)
        (reference_out_a.square().mean() + reference_out_b.abs().mean()).backward()

        sampler = flow.get_sampler(0)
        out_a = sampler(points_a)
        out_b = sampler(points_b)
        # Two independent backwards through the one cached sampler, WITHOUT
        # retain_graph: the shared pinned+scaled field graphs are cut at
        # detached leaves, so neither backward touches the other's graph.
        out_a.square().mean().backward()
        out_b.abs().mean().backward()
        flow.apply_accumulated_field_grad()

        torch.testing.assert_close(out_a, reference_out_a)
        torch.testing.assert_close(out_b, reference_out_b)
        torch.testing.assert_close(points_a.grad, reference_a.grad)
        torch.testing.assert_close(points_b.grad, reference_b.grad)
        torch.testing.assert_close(flow.flows[0].grad, reference_lr.grad)
        torch.testing.assert_close(flow.flows[1].grad, reference_hr.grad)
        self.assertIsNone(flow._pending_field_graphs)


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
