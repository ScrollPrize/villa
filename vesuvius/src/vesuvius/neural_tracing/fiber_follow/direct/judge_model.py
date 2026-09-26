"""Independent native-resolution image memory and prefix-query decoder."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

JUDGE_ARCHITECTURE = 'ct_slice_judge_v1'


@dataclass(frozen=True)
class JudgeConfig:
    pixels: int = 257
    spacing: float = .5
    widths: tuple = (16, 32, 64)
    strides: tuple = (1, 2, 2)
    full_grid: int = 8
    center_grid: int = 5
    center_spacing: float = 1.
    width: int = 128
    heads: int = 4
    layers: int = 2
    feedforward: int = 256
    view_batch: int = 16

    def __post_init__(self):
        if tuple(self.strides) != (1, 2, 2) or self.center_grid != 5 or self.pixels % 2 != 1:
            raise ValueError('Unsupported judge convolution/token lattice')
        if min(self.spacing, self.view_batch) <= 0 or (self.pixels-1)*self.spacing/2 < 2*self.center_spacing:
            raise ValueError('Center tokens must lie inside the native view')

    def to_dict(self):
        return asdict(self)


def sequence_tensors(records, device='cpu', anchor=0.):
    """Relative-only metadata. No annotation, absolute coordinates or fiber IDs."""
    last = records[-1]
    metadata = []
    for r in records:
        metadata.append(np.r_[(r['center']-last['center']) @ last['frame']/128,
                             (last['frame'].T @ r['frame']).ravel(),
                             (r['arc']-last['arc'])/128, (r['arc']-anchor)/128,
                             r['reference'], r['support'], r['query']])
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)[None]
    return dict(images=tensor([r['images'] for r in records]), metadata=tensor(metadata),
                valid=tensor([r['support'] for r in records], torch.bool),
                queries=tensor([r['query'] for r in records], torch.bool))


class CTJudge(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or JudgeConfig()
        cfg = self.cfg
        def stage(a, b, stride):
            return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU())
        self.local = stage(3, cfg.widths[0], 1)
        self.deep = nn.Sequential(stage(cfg.widths[0], cfg.widths[1], 2), stage(cfg.widths[1], cfg.widths[2], 2))
        self.full_projection = nn.Linear(cfg.widths[2], cfg.width)
        self.center_projection = nn.Linear(cfg.widths[0], cfg.width)
        self.view_embedding = nn.Embedding(3, cfg.width)
        self.group_embedding = nn.Embedding(2, cfg.width)
        self.position = nn.Linear(2, cfg.width)
        self.metadata = nn.Linear(17, cfg.width)
        self.query = nn.Linear(17, cfg.width)
        layer = nn.TransformerDecoderLayer(cfg.width, cfg.heads, cfg.feedforward,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers, nn.LayerNorm(cfg.width))
        self.head = nn.Linear(cfg.width, 1)
        center = torch.arange(-2, 3).float()*cfg.center_spacing
        yy, xx = torch.meshgrid(center, center, indexing='ij')
        self.register_buffer('center_coordinates', torch.stack((xx, yy), -1).reshape(-1, 2))
        lattice = torch.arange((cfg.pixels+3)//4).float()*4*cfg.spacing-(cfg.pixels-1)*cfg.spacing/2
        yy, xx = torch.meshgrid(lattice, lattice, indexing='ij')
        coordinates = F.adaptive_avg_pool2d(torch.stack((xx, yy))[None], cfg.full_grid)[0].flatten(1).T
        self.register_buffer('full_coordinates', coordinates)

    @property
    def tokens_per_view(self):
        return self.cfg.full_grid**2+25

    def encode(self, images):
        shape = images.shape[:-3]
        flat = images.reshape(-1, 3, self.cfg.pixels, self.cfg.pixels)
        tokens = []
        for part in flat.split(self.cfg.view_batch):
            tokens.append(self.encode_views(part))
        return torch.cat(tokens).reshape(*shape, self.tokens_per_view, self.cfg.width)

    def encode_views(self, images):
        """One bounded CNN batch; compiled independently of the sequence length."""
        local = self.local(images.contiguous(memory_format=torch.channels_last))
        full = F.adaptive_avg_pool2d(self.deep(local), self.cfg.full_grid).flatten(2).transpose(1, 2)
        grid = (self.center_coordinates/((self.cfg.pixels-1)*self.cfg.spacing/2)).reshape(1, 5, 5, 2)
        center = F.grid_sample(local.float(), grid.expand(len(images), -1, -1, -1), align_corners=True).flatten(2).transpose(1, 2)
        return torch.cat((self.full_projection(full), self.center_projection(center)), 1)

    def decode(self, tokens, metadata, valid, queries):
        b, n = metadata.shape[:2]
        device = tokens.device
        coords = torch.cat((self.full_coordinates, self.center_coordinates))
        groups = torch.cat((torch.zeros(self.cfg.full_grid**2, device=device, dtype=torch.long),
                            torch.ones(25, device=device, dtype=torch.long)))
        memory = tokens+self.position(coords)[None, None, None]+self.group_embedding(groups)[None, None, None]
        memory = memory+self.view_embedding(torch.arange(3, device=device))[None, None, :, None]
        memory = memory+self.metadata(metadata)[:, :, None, None]
        memory = memory.reshape(b, -1, self.cfg.width)
        padding = ~valid[:, :, None, None].expand(b, n, 3, self.tokens_per_view).reshape(b, -1)
        # A masked zero sentinel prevents all-masked attention NaNs; it has no evidence eligibility.
        memory = torch.cat((memory, torch.zeros(b, 1, self.cfg.width, device=device, dtype=memory.dtype)), 1)
        padding = torch.cat((padding, torch.zeros(b, 1, device=device, dtype=torch.bool)), 1)
        qpad = ~queries.clone()
        qpad[:, 0] = False
        decoded = self.decoder(self.query(metadata), memory, tgt_key_padding_mask=qpad, memory_key_padding_mask=padding)
        return self.head(decoded).squeeze(-1).float()

    def forward(self, images, metadata, valid, queries):
        return self.decode(self.encode(images), metadata, valid, queries)


class FeatureCache:
    """Fixed-weight cache. Relative metadata is always refreshed by the caller."""
    def __init__(self):
        self.identity = None
        self.entries = {}
        self.encoded_views = 0

    @torch.no_grad()
    def tokens(self, model, records, device):
        identity = (id(model), tuple(p._version for p in model.parameters()))
        if identity != self.identity:
            self.entries.clear()
            self.identity = identity
        keys = {r['key'] for r in records}
        self.entries = {k: v for k, v in self.entries.items() if k in keys}
        for r in records:
            if r['key'] not in self.entries:
                images = torch.as_tensor(r['images'], device=device)[None]
                self.entries[r['key']] = model.encode(images)[0]
                self.encoded_views += 3
        return torch.stack([self.entries[r['key']] for r in records])[None]
