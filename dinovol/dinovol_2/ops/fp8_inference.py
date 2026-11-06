"""Opt-in, forward-only FP8 transformer inference; checkpoint weights stay unchanged.

Linear layers use per-row E4M3 scales and fast-accumulation GEMMs. Attention uses
per-head scales, an FP32 online softmax/accumulator, and a BF16 result. This is
lossy computation, independent of the embedding output's int8 storage format.
"""
import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def require_supported_device(device):
    device = torch.device(device)
    if (triton is None or device.type != "cuda" or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) < (8, 9)):
        raise ValueError("FP8 inference requires Triton and an NVIDIA CUDA GPU with compute capability >= 8.9")


class FP8Linear(nn.Module):
    """Frozen inference replacement, installed only after loading the checkpoint."""

    def __init__(self, linear):
        super().__init__()
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 linear dimensions must be multiples of 16")
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448
        self.register_buffer("weight_fp8", (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn).t())
        self.register_buffer("weight_scale", scale.t().contiguous())
        self.register_buffer("bias", linear.bias.detach() if linear.bias is not None else None)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.eval()

    def forward(self, x):
        if self.training or (torch.is_grad_enabled() and x.requires_grad):
            raise RuntimeError("FP8Linear is inference-only")
        flat = x.reshape(-1, x.shape[-1]).float()
        scale = flat.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        quantized = (flat / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(quantized, self.weight_fp8, scale_a=scale, scale_b=self.weight_scale,
                               out_dtype=torch.bfloat16, use_fast_accum=True)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*x.shape[:-1], self.out_features)


if triton is not None:
    @triton.jit
    def _attention(Q, K, V, QS, KS, VS, Out, N: tl.constexpr, D: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        head = tl.program_id(1)
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        channels = tl.arange(0, BD)
        q = tl.load(Q + head * N * D + rows[:, None] * D + channels[None, :],
                    (rows[:, None] < N) & (channels[None, :] < D), other=0.)
        scale = tl.load(QS + head) * tl.load(KS + head) * (D ** -0.5 * 1.4426950408889634)
        value_scale = tl.load(VS + head)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        accumulator = tl.full((BM, BD), 0, tl.float32)
        for block in range(tl.cdiv(N, BN)):
            positions = block * BN + cols
            k = tl.load(K + head * N * D + positions[None, :] * D + channels[:, None],
                        (positions[None, :] < N) & (channels[:, None] < D), other=0.)
            scores = tl.dot(q, k).to(tl.float32) * scale
            scores = tl.where(positions[None, :] < N, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            probabilities = tl.exp2(scores - new_maximum[:, None])
            correction = tl.exp2(maximum - new_maximum)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            accumulator = accumulator * correction[:, None]
            # V is stored transposed so the FP8 dot's right operand is column-major.
            v = tl.load(V + head * N * D + channels[None, :] * N + positions[:, None],
                        (positions[:, None] < N) & (channels[None, :] < D), other=0.)
            # Scaling the unnormalized probabilities uses E4M3's range without
            # rounding small softmax probabilities to zero prematurely.
            p = (probabilities * 448.).to(tl.float8e4nv)
            accumulator = tl.dot(p, v, accumulator)
            maximum = new_maximum
        result = accumulator * (value_scale / (denominator * 448.))[:, None]
        tl.store(Out + head * N * D + rows[:, None] * D + channels[None, :], result,
                 (rows[:, None] < N) & (channels[None, :] < D))


@torch.compile(dynamic=False)
def _quantize_attention(q, k, v):
    scales = [x.float().abs().amax((-1, -2), keepdim=True).clamp_min(1e-12) / 448 for x in (q, k, v)]
    q8 = (q / scales[0]).to(torch.float8_e4m3fn).contiguous()
    k8 = (k / scales[1]).to(torch.float8_e4m3fn).contiguous()
    v8 = (v / scales[2]).to(torch.float8_e4m3fn).transpose(-1, -2).contiguous()
    return q8, k8, v8, *scales


@torch.library.custom_op("dinovol::fp8_attention", mutates_args=())
def fp8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.dtype != torch.bfloat16
            or k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device
            or not 1 <= q.shape[-1] <= 128 or q.numel() == 0):
        raise ValueError("FP8 attention needs matching nonempty BF16 [B,H,N,D] inputs with D <= 128")
    require_supported_device(q.device)
    batch, heads, points, channels = q.shape
    inputs = _quantize_attention(q, k, v)
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _attention[(triton.cdiv(points, 128), batch * heads)](
        *inputs, out, points, channels, 128, 64, max(32, triton.next_power_of_2(channels)),
        num_warps=4, num_stages=2,
    )
    return out


@fp8_attention.register_fake
def _fake_attention(q, k, v):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


def enable_fp8_transformer(backbone):
    """Convert the frozen transformer only; CNN, normalization and heads keep BF16/FP32.

    Validation precedes mutation. Models with axial attention or fused separate
    QKV biases are rejected instead of silently taking a different precision path.
    """
    from dinovol_2.model.dinov2_eva import EvaAttention

    require_supported_device(next(backbone.parameters()).device)
    if backbone.training or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("freeze and evaluate the backbone before enabling FP8")
    attentions = [m for m in backbone.blocks.modules() if isinstance(m, EvaAttention)]
    if not attentions or any(type(m) is not EvaAttention or m.q_bias is not None for m in attentions):
        raise ValueError("FP8 inference currently supports full EVA attention without separate QKV biases")
    for attention in attentions:
        projection = attention.qkv if attention.qkv is not None else attention.q_proj
        head_dim = projection.out_features // attention.num_heads // (3 if attention.qkv is not None else 1)
        if head_dim > 128:
            raise ValueError("FP8 attention supports head dimensions up to 128")
    replacements = [(parent, name, child) for parent in backbone.blocks.modules()
                    for name, child in parent.named_children() if isinstance(child, nn.Linear)]
    if any(child.in_features % 16 or child.out_features % 16 for _, _, child in replacements):
        raise ValueError("FP8 transformer linear dimensions must be multiples of 16")
    for parent, name, child in replacements:
        setattr(parent, name, FP8Linear(child))
    for attention in attentions:
        attention.inference_precision = "fp8"
    backbone.grad_checkpointing = False
    return len(replacements)
