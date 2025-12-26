import torch
from torch import nn

from timm.layers.pos_embed_sincos import apply_rot_embed_cat

from vesuvius.models.build.transformers.eva import Eva


class _DummyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)


class _DummyMlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc2 = nn.Linear(dim, dim, bias=False)


class CaptureBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, **_kwargs):
        super().__init__()
        self.attn = _DummyAttn(dim)
        self.mlp = _DummyMlp(dim)
        self.last_rope = None

    def forward(self, x, rope=None, attn_mask=None):
        self.last_rope = rope
        return x


def _make_eva_3d(block_fn):
    return Eva(
        embed_dim=96,
        depth=1,
        num_heads=4,
        ref_feat_shape=(2, 3, 4),
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        pos_emb_type="rope",
        block_fn=block_fn,
    )


def test_rope_3d_embed_shapes_types_values():
    eva = _make_eva_3d(block_fn=CaptureBlock)
    seq_len = 2 * 3 * 4
    x = torch.zeros(1, seq_len, eva.embed_dim, dtype=torch.float32)

    _, rot_pos_embed, keep_indices = eva._pos_embed(x)

    assert keep_indices is None
    assert eva.effective_rope_dim == eva.head_dim
    assert rot_pos_embed.shape == (seq_len, eva.head_dim * 2)
    assert rot_pos_embed.dtype == x.dtype

    sin_emb, cos_emb = rot_pos_embed.chunk(2, dim=-1)
    torch.testing.assert_close(sin_emb[0], torch.zeros_like(sin_emb[0]))
    torch.testing.assert_close(cos_emb[0], torch.ones_like(cos_emb[0]))
    torch.testing.assert_close(
        sin_emb.pow(2) + cos_emb.pow(2),
        torch.ones_like(sin_emb),
        atol=1e-5,
        rtol=1e-5,
    )

    q = torch.randn(1, seq_len, eva.head_dim)
    q_rot = apply_rot_embed_cat(q, rot_pos_embed)
    torch.testing.assert_close(q_rot[:, 0, :], q[:, 0, :], atol=1e-5, rtol=1e-5)
    assert q_rot.shape == q.shape


def test_rope_3d_passed_to_blocks_with_dtype():
    eva = _make_eva_3d(block_fn=CaptureBlock).double()
    seq_len = 2 * 3 * 4
    x = torch.zeros(1, seq_len, eva.embed_dim, dtype=torch.float64)

    eva.forward_features(x)

    block = eva.blocks[0]
    assert block.last_rope is not None
    assert block.last_rope.dtype == x.dtype
    assert block.last_rope.shape == (seq_len, eva.head_dim * 2)
