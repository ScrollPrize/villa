"""Combined cumulative-width coordinates with explicit, deterministic gradients.

The boundary input MUST be [0, cumsum(widths)]. Its derivative is incorporated
in the returned width derivative, not returned separately through boundaries.
CUDA FP32 uses tiled Triton reductions (no atomics); CPU/double is a small-test
reference. First derivatives only. No loss or hypothesis approximation.
"""
import torch


class _Coordinates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, d, widths, boundaries, division):
        ctx.division = division
        s = a.floor().long().clamp(0, widths.numel() - 1)
        u = boundaries[s] + (a - s) * widths[s] + d
        t = (torch.searchsorted(boundaries.detach(), u.contiguous(), right=True)
             - 1).clamp(0, widths.numel() - 1)
        q = t + (u - boundaries[t]) / widths[t]
        # Exact cancellation when the offset remains within the anchor bucket.
        q = torch.where(s == t, a + d / widths[s], q)
        ctx.save_for_backward(a, d, widths, s, t, q)
        return u, q

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gu, gq):
        a, d, w, s, t, q = ctx.saved_tensors
        gu = torch.zeros_like(a) if gu is None else gu.contiguous()
        gq = torch.zeros_like(a) if gq is None else gq.contiguous()
        if a.is_cuda and a.dtype == torch.float32:
            from .fiber_uv_coordinates_cuda import backward
            ga, gd, gw = backward(a, d, w, s, t, q, gu, gq, ctx.division)
        else:
            # Deterministic dense reference, only for tests/small CPU inputs.
            g = gu + gq / w[t]
            ga, gd = g * w[s], g
            j = torch.arange(w.numel(), device=w.device)
            sf, tf = s.reshape(-1, 1), t.reshape(-1, 1)
            h = (gq / w[t]).reshape(-1, 1)
            prefix = gu.reshape(-1, 1) * (j < sf)
            cross = h * ((j < sf).to(a.dtype) - (j < tf).to(a.dtype))
            cross = cross + (g * (a-s)).reshape(-1, 1) * (j == sf)
            cross = cross - (gq / w[t] * (q-t)).reshape(-1, 1) * (j == tf)
            same = (gu * (a-s) - gq*d/w[s].square()).reshape(-1, 1) * (j == sf)
            gw = (prefix + torch.where(sf == tf, same, cross)).sum(0)
        return ga, gd, gw, None, None


def combined_coordinates(a, d, widths, boundaries, division='standard'):
    """Map anchor turn + physical distance to (physical U, endpoint turn)."""
    if division not in ('standard', 'precise'):
        raise ValueError('Unknown coordinate division: '+str(division))
    return _Coordinates.apply(a.contiguous(), d.expand_as(a).contiguous(), widths, boundaries, division)


def validate_cuda(division='standard'):
    """Managed-run preflight: tail tiles, extrapolation and repeated gradients."""
    from .fiber_uv_layout import q_to_u, u_to_q
    torch.manual_seed(31)
    a=(torch.rand(33,151,device='cuda')*155-2).requires_grad_()
    d=(torch.randn(33,1,device='cuda')*500).requires_grad_()
    w=(torch.rand(151,device='cuda')*100+100).requires_grad_()
    gu,gq=torch.randn_like(a)*.01,torch.randn_like(a)
    out=[]
    for combined in [False,True,True]:
        c=torch.cat((w.new_zeros(1),w.cumsum(0)))
        if combined:u,q=combined_coordinates(a,d,w,c,division)
        else:
            u=q_to_u(a,w,c)+d
            q=u_to_q(u,w,c)
        out.append((u,q,*torch.autograd.grad((u*gu+q*gq).sum(),(a,d,w))))
    for old,new in zip(out[0],out[1]):
        torch.testing.assert_close(old,new,rtol=3e-4,atol=1e-4)
    for first,repeat in zip(out[1],out[2]):
        torch.testing.assert_close(first,repeat,rtol=0,atol=0)
    return {'value_gradient_check': 'passed', 'repeat_bitwise': True, 'division': division}
