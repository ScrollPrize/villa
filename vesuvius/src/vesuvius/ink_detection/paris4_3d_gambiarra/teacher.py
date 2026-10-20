"""The DINO teacher: dinovol_2 ViT backbone (patch 8, 864 channels, 24 blocks) + linear point readout, plus (stages
2-4) a learned voxel residual.

Point logit at fractional ZYX view coordinate q (voxels of the 128^3 view):
    head(normalize(sample_normalized_patch_embeddings(tokens, q)))  [+ voxel_sample(residual(tokens, view), q)]
The backbone runs under bf16 autocast; the readout, residual and head run in float32.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

SSL_CHECKPOINT_SHA256 = '5fb1cf4bd831275bdea28ba522ee76641aa1e42f194a61a7fdb25b0c6670083a'


def load_backbone(path):
    """teacher.backbone of the dinovol_2 SSL checkpoint (HF scrollprize/dinovol_v2_ps8_with_paris4_352500), on CUDA."""
    from .runtime import sha256_file
    if sha256_file(path) != SSL_CHECKPOINT_SHA256:
        raise ValueError('SSL checkpoint is not dinovol_v2_ps8_with_paris4_352500 (sha256 mismatch)')
    from dinovol_2.model.model import DinoVitStudentTeacher
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = dict(state['config']['model'])
    cfg.update(pretrained_weights=str(path), pretrained_backbone_only=False)
    backbone = DinoVitStudentTeacher(cfg).teacher.backbone
    expected = {k.removeprefix('backbone.'): v for k, v in state['teacher'].items() if k.startswith('backbone.')}
    actual = backbone.state_dict()
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], v) for k, v in expected.items()), \
        'loaded backbone differs from the checkpoint teacher tensors'
    assert len(backbone.blocks) == 24
    return backbone.eval().requires_grad_(False).cuda()


def upsample2(x):
    """Separable x2 linear resize (align_corners=False) by index_select, deterministic in backward."""
    for axis in (2, 3, 4):
        size = x.shape[axis]
        q = ((torch.arange(size * 2, device=x.device, dtype=torch.float32) + .5) / 2 - .5).clamp(0, size - 1)
        low = q.floor().long()
        high = (low + 1).clamp_max(size - 1)
        shape = [1] * 5
        shape[axis] = size * 2
        w = (q - low).reshape(shape)
        x = x.index_select(axis, low) * (1 - w) + x.index_select(axis, high) * w
    return x


def voxel_sample(volume, q):
    """Trilinear value of volume [1, 1, D, H, W] at fractional ZYX voxel-centre coordinates q [N, 3] (clamped to
    the volume): explicit gather, deterministic in backward."""
    assert volume.ndim == 5 and volume.shape[:2] == (1, 1)
    shape = q.new_tensor(volume.shape[-3:])
    q = q.clamp_min(0).minimum(shape - 1)
    lower = q.floor().long()
    upper = (lower + 1).minimum(shape.long() - 1)
    fraction = q - lower
    flat = volume[0, 0].reshape(-1)
    answer = torch.zeros(len(q), device=q.device, dtype=volume.dtype)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                bits = (z, y, x)
                idx = [upper[:, j] if bits[j] else lower[:, j] for j in range(3)]
                weights = [fraction[:, j] if bits[j] else 1 - fraction[:, j] for j in range(3)]
                answer = answer + flat[(idx[0] * volume.shape[-2] + idx[1]) * volume.shape[-1] + idx[2]] * weights[0] * weights[1] * weights[2]
    return answer


class SpatialResidual(nn.Module):
    """16^3 tokens -> 128^3 logit residual: 1x1 projection to 32 channels, four 3^3 convolutions with x2 upsampling
    between them (32-32-16-8-4), plus a two-layer CT branch on the view; zero-initialized output."""
    def __init__(self, channels=864):
        super().__init__()
        self.project = nn.Conv3d(channels, 32, 1)
        self.blocks = nn.ModuleList([nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate'),
                                     nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate'),
                                     nn.Conv3d(16, 8, 3, padding=1, padding_mode='replicate'),
                                     nn.Conv3d(8, 4, 3, padding=1, padding_mode='replicate')])
        self.output = nn.Conv3d(4, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.ct = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1, padding_mode='replicate'), nn.GELU(),
                                nn.Conv3d(4, 4, 3, padding=1, padding_mode='replicate'), nn.GELU())

    def forward(self, tokens, image):
        assert tokens.shape[1] == 4096 and image.shape[1:] == (1, 128, 128, 128)
        h = F.normalize(tokens.float(), dim=-1).transpose(1, 2).reshape(len(tokens), -1, 16, 16, 16)
        h = F.gelu(self.project(h))
        for j, layer in enumerate(self.blocks):
            if j:
                h = upsample2(h)
            h = F.gelu(layer(h))
        h = h + self.ct(image.float())
        return self.output(h)


class Teacher(nn.Module):
    """residual=False: stage 1 (backbone + linear head). residual=True: stages 2-4.

    Parameter names: backbone.*, head.*, residual.*. The original checkpoints name the first two native.backbone.* /
    native.head.* (stages 2-4); convert_original() maps them."""
    def __init__(self, backbone, residual, checkpoint_mode, seed=27):
        super().__init__()
        assert checkpoint_mode in ('all', 'half', 'none')
        self.backbone = backbone
        self.backbone.requires_grad_(True)
        self.backbone.mask_token.requires_grad_(False)  # no masks are used: no forward contribution
        self.backbone.eval()  # no dropout / drop-path
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(864, 1)
            nn.init.normal_(self.head.weight, std=.01)
            nn.init.zeros_(self.head.bias)
        self.head.cuda()
        self.residual = None
        if residual:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.residual = SpatialResidual().cuda()
        self.checkpoint_mode = checkpoint_mode
        self.backbone_active = True  # stages 2: False for the first 250 updates (decoder warm-up)

    def _tokens(self, x):
        b = self.backbone
        grad = torch.is_grad_enabled() and self.checkpoint_mode != 'none'
        if grad:
            v, rope, shape = checkpoint(b.prepare_tokens_with_masks, x, None, view_kind='global', use_reentrant=False)
        else:
            v, rope, shape = b.prepare_tokens_with_masks(x, None, view_kind='global')
        coords = b._get_shared_per_block_rope_coords(shape)
        for index, block in enumerate(b.blocks):
            if grad and (self.checkpoint_mode == 'all' or index % 2 == 0):
                v = checkpoint(block, v, rope=rope, rope_shape=shape, rope_coords=coords, use_reentrant=False)
            else:
                v = block(v, rope=rope, rope_shape=shape, rope_coords=coords)
        return b.norm(v)[:, b.num_prefix_tokens:]

    def tokens(self, x):
        """[B, 4096, 864] patch tokens of views x [B, 1, 128, 128, 128] (bf16 autocast). Without gradients when
        the backbone is not active (frozen-backbone warm-up) or under no_grad."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.backbone_active), torch.autocast('cuda', dtype=torch.bfloat16):
            return self._tokens(x)

    def readout(self, tokens, q, index=0):
        """Linear-head logits at q [N, 3] of view `index` of tokens [B, 4096, 864]."""
        from dinovol_2.ops.point_embeddings import sample_normalized_patch_embeddings
        q = q.float()
        z = sample_normalized_patch_embeddings(tokens, torch.full((len(q),), index, device=q.device, dtype=torch.long), q, (8,) * 3, (16,) * 3)
        return self.head(F.normalize(z.float(), dim=-1).float()).squeeze(-1)

    def field(self, tokens, x):
        return self.residual(tokens, x)

    def query(self, tokens, field, q):
        logits = self.readout(tokens, q)
        return logits if field is None else logits + voxel_sample(field, q.float())

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        assert set(params) == set(state), sorted(set(params) ^ set(state))[:8]
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(state[name])

    @torch.no_grad()
    def targets(self, x, sites, microbatch=2, chunk=4096):
        """Frozen-teacher sigmoid targets: tokens for `microbatch` views at a time, the residual field per view in
        float32, logits at each view's sites [N_j, 3] in chunks of `chunk` sites. The original FastTeacher producer
        selected Flash Attention explicitly; do the same through dinovol, without copying its attention code."""
        out = []
        for start in range(0, len(x), microbatch):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                tokens = self.tokens(x[start:start + microbatch])
            for k in range(len(tokens)):
                j = start + k
                field = self.field(tokens[k:k + 1], x[j:j + 1]) if self.residual is not None else None
                q = torch.as_tensor(sites[j], device=x.device)
                out.append(torch.cat([self.query(tokens[k:k + 1], field, part) for part in q.split(chunk)]).sigmoid())
        return out


def convert_original(trainable):
    """Original checkpoint 'delta' / 'trainable' names -> this module's names."""
    return {k.replace('native.backbone.', 'backbone.').replace('native.head.', 'head.'): v for k, v in trainable.items()}
