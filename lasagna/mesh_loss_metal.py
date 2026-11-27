"""Fused Metal mesh regularizers for MPS tensors.

On MPS each PyTorch op of a mesh loss is its own GPU dispatch, and a fit step runs hundreds of them on tiny
tensors, forward and backward; on a 10k-node mesh the dispatches cost far more than the arithmetic. These kernels
compute a whole loss map in one dispatch and its gradient in one more, with the arithmetic of the PyTorch
references (``opt_loss_step.step_loss_maps``, ``opt_loss_bend``, ``opt_loss_smooth``) in the same float32 order where the order is defined.

The backward is a gather (each vertex sums the contributions of its up to four quads), so it is deterministic and
needs no float atomics. Compiled once per process with ``torch.mps.compile_shader`` and encoded on PyTorch's
current MPS stream.
"""
from __future__ import annotations

import os

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float3 ld(device const float* x, int d, int i, int j, int H, int W) {
    ulong o = ((ulong(d) * ulong(H) + ulong(i)) * ulong(W) + ulong(j)) * 3;
    return float3(x[o], x[o + 1], x[o + 2]);
}

// torch: valid = |v| > 1e-6, unit = valid ? v / max(|v|, 1e-12) : 0
inline float3 unit_dir(float3 v, thread float& valid) {
    float l = precise::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    valid = l > 1.0e-6f ? 1.0f : 0.0f;
    return valid > 0.0f ? v / max(l, 1.0e-12f) : float3(0.0f);
}

// _normalize_direction of a summed direction with its valid count
inline float3 norm_dir(float3 s, float count, thread bool& valid) {
    s = count > 0.0f ? s : float3(0.0f);
    float l = precise::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    valid = l > 1.0e-6f;
    return valid ? s / max(l, 1.0e-12f) : float3(0.0f);
}

// _directional_step_penalty for one edge: returns the penalty and writes d(penalty)/d(diff)
inline float step_penalty(float3 diff, float t, float tsq, float3 dir, bool valid, thread float3& grad) {
    float L = precise::sqrt((diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) + 1.0e-8f);
    if (valid) {
        float pd = diff.x * dir.x + diff.y * dir.y + diff.z * dir.z;
        float a = max(t - fabs(pd), 0.0f);
        float b = max(L - t, 0.0f);
        // abs and relu have zero gradient at zero in torch
        float sgn = pd > 0.0f ? 1.0f : (pd < 0.0f ? -1.0f : 0.0f);
        grad = (-2.0f * a / tsq) * sgn * dir + (2.0f * b / tsq) * (diff / L);
        return a * a / tsq + b * b / tsq;
    }
    float rel = (L - t) / t;
    grad = (2.0f * rel / t) * (diff / L);
    return rel * rel;
}

// one thread per quad (d, i, j); tp: t, t^2, t_diag, t_diag^2 (rounded on the host as the reference does)
kernel void step_fwd(device const float* x [[buffer(0)]],
                     device float* lm [[buffer(1)]],
                     device float* jac [[buffer(2)]],
                     constant float* tp [[buffer(3)]],
                     constant long* dims [[buffer(4)]],   // D, H, W
                     uint q [[thread_position_in_grid]]) {
    int D = int(dims[0]), H = int(dims[1]), W = int(dims[2]);
    int Hq = H - 1, Wq = W - 1;
    if (long(q) >= long(D) * Hq * Wq) return;
    int j = int(q % uint(Wq)), i = int((q / uint(Wq)) % uint(Hq)), d = int(q / uint(Wq * Hq));

    // H edges of row i at columns j-1, j, j+1; the direction averages them (3 taps only when W > 2)
    float vh0, vh1, vhm = 0.0f;
    float3 dh = ld(x, d, i + 1, j, H, W) - ld(x, d, i, j, H, W);
    float3 uh = unit_dir(dh, vh0) + unit_dir(ld(x, d, i + 1, j + 1, H, W) - ld(x, d, i, j + 1, H, W), vh1);
    float ch = vh0 + vh1;
    if (W > 2 && j >= 1) {
        uh = uh + unit_dir(ld(x, d, i + 1, j - 1, H, W) - ld(x, d, i, j - 1, H, W), vhm);
        ch = ch + vhm;
    }
    bool valid_h;
    float3 dir_h = norm_dir(uh, ch, valid_h);

    // W edges of column j at rows i-1, i, i+1 (3 taps only when H > 2)
    float vw0, vw1, vwm = 0.0f;
    float3 dw = ld(x, d, i, j + 1, H, W) - ld(x, d, i, j, H, W);
    float3 uw = unit_dir(dw, vw0) + unit_dir(ld(x, d, i + 1, j + 1, H, W) - ld(x, d, i + 1, j, H, W), vw1);
    float cw = vw0 + vw1;
    if (H > 2 && i >= 1) {
        uw = uw + unit_dir(ld(x, d, i - 1, j + 1, H, W) - ld(x, d, i - 1, j, H, W), vwm);
        cw = cw + vwm;
    }
    bool valid_w;
    float3 dir_w = norm_dir(uw, cw, valid_w);

    float vd1, vd2;
    float3 dd1 = ld(x, d, i + 1, j + 1, H, W) - ld(x, d, i, j, H, W);
    float3 dir_d1 = unit_dir(dd1, vd1);
    float3 dd2 = ld(x, d, i + 1, j, H, W) - ld(x, d, i, j + 1, H, W);
    float3 dir_d2 = unit_dir(dd2, vd2);

    float3 g_h, g_w, g_d1, g_d2;
    float p = step_penalty(dh, tp[0], tp[1], dir_h, valid_h, g_h);
    p = p + step_penalty(dw, tp[0], tp[1], dir_w, valid_w, g_w);
    p = p + step_penalty(dd1, tp[2], tp[3], dir_d1, vd1 > 0.0f, g_d1);
    p = p + step_penalty(dd2, tp[2], tp[3], dir_d2, vd2 > 0.0f, g_d2);
    lm[q] = p * 0.25f;
    device float* J = jac + ulong(q) * 12;
    float3 c[4] = {0.25f * g_h, 0.25f * g_w, 0.25f * g_d1, 0.25f * g_d2};
    for (int k = 0; k < 4; ++k) {
        J[3 * k + 0] = c[k].x;
        J[3 * k + 1] = c[k].y;
        J[3 * k + 2] = c[k].z;
    }
}

inline float3 jv(device const float* J, int k) { return float3(J[3 * k], J[3 * k + 1], J[3 * k + 2]); }

// one thread per vertex: sum the edge gradients of the up-to-four quads that contain it
// edges per quad: h (i,j)->(i+1,j), w (i,j)->(i,j+1), d1 (i,j)->(i+1,j+1), d2 (i,j+1)->(i+1,j)
kernel void step_bwd(device const float* g [[buffer(0)]],
                     device const float* jac [[buffer(1)]],
                     device float* gx [[buffer(2)]],
                     constant long* dims [[buffer(3)]],   // D, H, W
                     uint v [[thread_position_in_grid]]) {
    int D = int(dims[0]), H = int(dims[1]), W = int(dims[2]);
    int Hq = H - 1, Wq = W - 1;
    if (long(v) >= long(D) * H * W) return;
    int vj = int(v % uint(W)), vi = int((v / uint(W)) % uint(H)), d = int(v / uint(W * H));
    float3 acc = float3(0.0f);
    for (int ri = 0; ri <= 1; ++ri) {
        int qi = vi - ri;
        if (qi < 0 || qi >= Hq) continue;
        for (int rj = 0; rj <= 1; ++rj) {
            int qj = vj - rj;
            if (qj < 0 || qj >= Wq) continue;
            ulong q = (ulong(d) * ulong(Hq) + ulong(qi)) * ulong(Wq) + ulong(qj);
            float gq = g[q];
            device const float* J = jac + q * 12;
            if (ri == 0 && rj == 0) acc -= gq * (jv(J, 0) + jv(J, 1) + jv(J, 2));
            if (ri == 1 && rj == 0) acc += gq * (jv(J, 0) + jv(J, 3));
            if (ri == 0 && rj == 1) acc += gq * (jv(J, 1) - jv(J, 3));
            if (ri == 1 && rj == 1) acc += gq * jv(J, 2);
        }
    }
    ulong o = ulong(v) * 3;
    gx[o] = acc.x;
    gx[o + 1] = acc.y;
    gx[o + 2] = acc.z;
}
// interior 3x3 stencils (bend, smooth): one output per interior vertex (i, j), 1 <= i <= H-2, 1 <= j <= W-2,
// stored at (d, i-1, j-1) of a (D, H-2, W-2) map. jac holds d(output)/d(vertex) for the 9 vertices around the
// centre, 27 floats per output, role (di+1)*3 + (dj+1)
inline void put(device float* J, int di, int dj, float3 g) {
    int r = 3 * ((di + 1) * 3 + (dj + 1));
    J[r] += g.x;
    J[r + 1] += g.y;
    J[r + 2] += g.z;
}

// bend penalty of one triplet prev -> mid -> next; adds its gradient (times w) to the three roles
inline float bend_pair(float3 e1, float3 e2, float thr, float w, device float* J, int pi, int pj, int ni, int nj) {
    float r1 = precise::rsqrt((e1.x * e1.x + e1.y * e1.y + e1.z * e1.z) + 1.0e-12f);
    float r2 = precise::rsqrt((e2.x * e2.x + e2.y * e2.y + e2.z * e2.z) + 1.0e-12f);
    float3 n1 = e1 * r1, n2 = e2 * r2;
    float3 m = -n1 * n2;
    float c = m.x + m.y + m.z;
    float ex = max(c - thr, 0.0f);
    if (ex > 0.0f) {
        // d(e * rsqrt(|e|^2 + eps))^T u = r u - r^3 e (e . u), with u = d(cos)/d(unit edge)
        float3 u1 = -n2, u2 = -n1;
        float3 g1 = (2.0f * ex * w) * (r1 * u1 - (r1 * r1 * r1) * dot(e1, u1) * e1);
        float3 g2 = (2.0f * ex * w) * (r2 * u2 - (r2 * r2 * r2) * dot(e2, u2) * e2);
        // e1 = mid - prev, e2 = next - mid
        put(J, pi, pj, -g1);
        put(J, 0, 0, g1 - g2);
        put(J, ni, nj, g2);
    }
    return ex * ex;
}

kernel void bend_fwd(device const float* x [[buffer(0)]],
                     device float* lm [[buffer(1)]],
                     device float* jac [[buffer(2)]],
                     constant float* sc [[buffer(3)]],   // cos threshold
                     constant long* dims [[buffer(4)]],   // D, H, W
                     uint q [[thread_position_in_grid]]) {
    int D = int(dims[0]), H = int(dims[1]), W = int(dims[2]);
    int Hi = H - 2, Wi = W - 2;
    if (long(q) >= long(D) * Hi * Wi) return;
    int j = int(q % uint(Wi)) + 1, i = int((q / uint(Wi)) % uint(Hi)) + 1, d = int(q / uint(Wi * Hi));
    float thr = sc[0];
    device float* J = jac + ulong(q) * 27;
    for (int k = 0; k < 27; ++k) J[k] = 0.0f;
    float3 c = ld(x, d, i, j, H, W);
    float ph = bend_pair(c - ld(x, d, i - 1, j, H, W), ld(x, d, i + 1, j, H, W) - c, thr, 0.25f, J, -1, 0, 1, 0);
    float pw = bend_pair(c - ld(x, d, i, j - 1, H, W), ld(x, d, i, j + 1, H, W) - c, thr, 0.25f, J, 0, -1, 0, 1);
    float p0 = bend_pair(c - ld(x, d, i - 1, j - 1, H, W), ld(x, d, i + 1, j + 1, H, W) - c, thr, 0.25f, J, -1, -1, 1, 1);
    float p1 = bend_pair(c - ld(x, d, i - 1, j + 1, H, W), ld(x, d, i + 1, j - 1, H, W) - c, thr, 0.25f, J, -1, 1, 1, -1);
    lm[q] = 0.25f * (((ph + pw) + p0) + p1);
}

kernel void smooth_fwd(device const float* x [[buffer(0)]],
                       device float* lm [[buffer(1)]],
                       device float* jac [[buffer(2)]],
                       constant float* sc [[buffer(3)]],   // mesh_step^2
                       constant long* dims [[buffer(4)]],   // D, H, W
                       uint q [[thread_position_in_grid]]) {
    int D = int(dims[0]), H = int(dims[1]), W = int(dims[2]);
    int Hi = H - 2, Wi = W - 2;
    if (long(q) >= long(D) * Hi * Wi) return;
    int j = int(q % uint(Wi)) + 1, i = int((q / uint(Wi)) % uint(Hi)) + 1, d = int(q / uint(Wi * Hi));
    float step_sq = sc[0];
    float3 avg = 0.25f * (((ld(x, d, i - 1, j, H, W) + ld(x, d, i + 1, j, H, W)) + ld(x, d, i, j - 1, H, W))
                          + ld(x, d, i, j + 1, H, W));
    float3 diff = ld(x, d, i, j, H, W) - avg;
    float3 sq = diff * diff;
    lm[q] = ((sq.x + sq.y) + sq.z) / step_sq;
    float3 g = (2.0f / step_sq) * diff;
    device float* J = jac + ulong(q) * 27;
    for (int k = 0; k < 27; ++k) J[k] = 0.0f;
    put(J, 0, 0, g);
    put(J, -1, 0, -0.25f * g);
    put(J, 1, 0, -0.25f * g);
    put(J, 0, -1, -0.25f * g);
    put(J, 0, 1, -0.25f * g);
}

// one thread per vertex: sum g * jac over the up-to-nine interior outputs whose 3x3 stencil contains it
kernel void stencil3_bwd(device const float* g [[buffer(0)]],
                         device const float* jac [[buffer(1)]],
                         device float* gx [[buffer(2)]],
                         constant long* dims [[buffer(3)]],   // D, H, W
                         uint v [[thread_position_in_grid]]) {
    int D = int(dims[0]), H = int(dims[1]), W = int(dims[2]);
    int Hi = H - 2, Wi = W - 2;
    if (long(v) >= long(D) * H * W) return;
    int vj = int(v % uint(W)), vi = int((v / uint(W)) % uint(H)), d = int(v / uint(W * H));
    float3 acc = float3(0.0f);
    for (int di = -1; di <= 1; ++di) {
        int ci = vi - di;
        if (ci < 1 || ci > Hi) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int cj = vj - dj;
            if (cj < 1 || cj > Wi) continue;
            ulong q = (ulong(d) * ulong(Hi) + ulong(ci - 1)) * ulong(Wi) + ulong(cj - 1);
            int r = 3 * ((di + 1) * 3 + (dj + 1));
            device const float* J = jac + q * 27;
            acc += g[q] * float3(J[r], J[r + 1], J[r + 2]);
        }
    }
    ulong o = ulong(v) * 3;
    gx[o] = acc.x;
    gx[o + 1] = acc.y;
    gx[o + 2] = acc.z;
}
"""

_lib = None
_GROUP = 256


def enabled(xyz: torch.Tensor, min_size: int) -> bool:
	"""Whether a mesh loss on ``xyz`` (D, H, W, 3) runs here: float32 MPS, H and W >= min_size, LASAGNA_METAL not 0."""
	# smaller meshes give empty maps, which the PyTorch path handles as before
	return (
		xyz.device.type == "mps"
		and xyz.dtype == torch.float32
		and int(xyz.shape[1]) >= min_size
		and int(xyz.shape[2]) >= min_size
		and os.environ.get("LASAGNA_METAL", "1") != "0"
	)


def _library():
	global _lib
	if _lib is None:
		_lib = torch.mps.compile_shader(_SOURCE)
	return _lib


def _launch(name: str, n: int, *args) -> None:
	if n > 0:
		getattr(_library(), name)(*args, threads=n, group_size=min(_GROUP, n))


def _rounded_target(value: float) -> tuple[float, float]:
	# the reference rounds the target and its square on a float32 host tensor (opt_loss_step._directional_step_penalty)
	host = torch.tensor(float(value), dtype=torch.float32).clamp_min(1.0e-12)
	return float(host), float(host.square())


class _StepLossMap(torch.autograd.Function):
	@staticmethod
	def forward(ctx, xyz: torch.Tensor, targets: list[float]) -> torch.Tensor:
		x = xyz.detach().to(torch.float32).contiguous()
		D, H, W, _ = (int(v) for v in x.shape)
		lm = torch.empty(D, 1, H - 1, W - 1, dtype=torch.float32, device=x.device)
		jac = torch.empty(D * (H - 1) * (W - 1) * 12, dtype=torch.float32, device=x.device)
		_launch("step_fwd", D * (H - 1) * (W - 1), x, lm, jac, targets, [D, H, W])
		ctx.save_for_backward(jac)
		ctx.shape = (D, H, W)
		ctx.dtype = xyz.dtype
		return lm

	@staticmethod
	def backward(ctx, grad_lm: torch.Tensor):
		(jac,) = ctx.saved_tensors
		D, H, W = ctx.shape
		# the grad of a mean arrives as an expanded (stride 0) view; the kernel reads dense memory
		g = grad_lm.to(torch.float32).contiguous()
		gx = torch.empty(D, H, W, 3, dtype=torch.float32, device=g.device)
		_launch("step_bwd", D * H * W, g, jac, gx, [D, H, W])
		return gx.to(ctx.dtype), None


def step_loss_map(xyz: torch.Tensor, mesh_step: float) -> torch.Tensor:
	"""``opt_loss_step.step_loss_maps``'s loss map, (D, 1, H-1, W-1), for an MPS (D, H, W, 3) mesh with H, W >= 2."""
	t, tsq = _rounded_target(mesh_step)
	td, tdsq = _rounded_target(float(mesh_step) * 2.0 ** 0.5)
	return _StepLossMap.apply(xyz, [t, tsq, td, tdsq])


class _Stencil3Map(torch.autograd.Function):
	"""A (D, 1, H-2, W-2) interior loss map from ``kernel`` with the shared 3x3 gather backward."""

	@staticmethod
	def forward(ctx, xyz: torch.Tensor, kernel: str, scalar: float) -> torch.Tensor:
		x = xyz.detach().to(torch.float32).contiguous()
		D, H, W, _ = (int(v) for v in x.shape)
		n = D * (H - 2) * (W - 2)
		lm = torch.empty(D, 1, H - 2, W - 2, dtype=torch.float32, device=x.device)
		jac = torch.empty(n * 27, dtype=torch.float32, device=x.device)
		# a one-element list: compile_shader binds lists of floats as float32 arrays
		_launch(kernel, n, x, lm, jac, [scalar], [D, H, W])
		ctx.save_for_backward(jac)
		ctx.shape = (D, H, W)
		ctx.dtype = xyz.dtype
		return lm

	@staticmethod
	def backward(ctx, grad_lm: torch.Tensor):
		(jac,) = ctx.saved_tensors
		D, H, W = ctx.shape
		g = grad_lm.to(torch.float32).contiguous()
		gx = torch.empty(D, H, W, 3, dtype=torch.float32, device=g.device)
		_launch("stencil3_bwd", D * H * W, g, jac, gx, [D, H, W])
		return gx.to(ctx.dtype), None, None


def _f32(value: float) -> float:
	# the reference applies Python scalars to float32 tensors, which rounds them to float32 first
	return float(torch.tensor(float(value), dtype=torch.float32))


def bend_loss_map(xyz: torch.Tensor, cos_threshold: float) -> torch.Tensor:
	"""``opt_loss_bend.bend_loss``'s map, (D, 1, H-2, W-2), for an MPS (D, H, W, 3) mesh with H, W >= 3."""
	return _Stencil3Map.apply(xyz, "bend_fwd", _f32(cos_threshold))


def smooth_loss_map(xyz: torch.Tensor, mesh_step: float) -> torch.Tensor:
	"""``opt_loss_smooth.smooth_loss``'s map, (D, 1, H-2, W-2), for an MPS (D, H, W, 3) mesh with H, W >= 3."""
	return _Stencil3Map.apply(xyz, "smooth_fwd", _f32(float(mesh_step) ** 2))
