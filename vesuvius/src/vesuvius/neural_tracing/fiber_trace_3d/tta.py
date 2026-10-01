"""Opt-in cubic TTA in canonical double-angle projection coordinates."""
from dataclasses import replace
from itertools import permutations, product
import json
import math
import time
from pathlib import Path

import torch

from .inference_adapter import FiberTrace3DPredictAdapter

IDENTITY = ((0, 1, 2), (False, False, False))
TRANSFORMS = tuple((p, IDENTITY[1]) for p in permutations(range(3))) + tuple(
    (IDENTITY[0], f) for f in product((False, True), repeat=3) if any(f))
VARIANTS = ('mean', 'median')
SCHEMA = 'double-angle-13-two-products-v2-strict-fp32'
PAIRS = ((0, 1), (0, 2), (1, 2))  # XYZ projections: XY, XZ, YZ.


def transform_input(tile, transform):
    p, flips = transform
    value = tile.permute(0, 1, *(2 + a for a in p))
    return torch.flip(value, [2 + a for a, yes in enumerate(flips) if yes]).contiguous()


def inverse_space(value, transform):
    p, flips = transform
    value = torch.flip(value, [2 + a for a, yes in enumerate(flips) if yes])
    return value.permute(0, 1, *(2 + p.index(a) for a in range(3)))


def reframe(raw, transform):
    """Undo space and map six encoded channels to canonical cos2/sin2."""
    raw = inverse_space(raw, transform)
    p, flips = transform
    xyz = tuple(2 - p[2 - a] for a in range(3))
    signs = tuple(-1 if flips[2 - a] else 1 for a in range(3))
    out = torch.empty_like(raw)
    for plane, (a, b) in enumerate(PAIRS):
        i, j = xyz.index(a), xyz.index(b)
        source = PAIRS.index(tuple(sorted((i, j))))
        c = 2 * raw[:, 2 * source] - 1
        s = c - math.sqrt(2) * (2 * raw[:, 2 * source + 1] - 1)
        out[:, 2 * plane] = c if i < j else -c
        out[:, 2 * plane + 1] = s * signs[i] * signs[j]
    out[:, 6] = raw[:, 6]
    return out


def encode_aggregate(values):
    """Project each averaged double-angle vector into its valid unit disk."""
    out = torch.empty_like(values)
    for plane in range(3):
        c, s = values[:, 2 * plane], values[:, 2 * plane + 1]
        norm = torch.sqrt(c.square() + s.square()).clamp_min(1)
        c, s = c / norm, s / norm
        out[:, 2 * plane] = .5 + .5 * c
        out[:, 2 * plane + 1] = .5 + .5 * (c - s) / math.sqrt(2)
    out[:, 6] = values[:, 6]
    return out.clamp(0, 1)


def midpoint_median(values):
    ordered = values.sort(dim=0).values
    n = len(ordered)
    return (ordered[(n - 1) // 2] + ordered[n // 2]) * .5


class TtaPredictAdapter(FiberTrace3DPredictAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.option_count != 1 or self.embedding_channels or self.native_output_stride != 1:
            raise ValueError('TTA pilot requires one direction branch, stride1 and no embeddings')
        original = self.output_products[0]
        self.option_count = len(VARIANTS)
        self._products = tuple(replace(original, name=f'fiber_{variant}', channels=tuple(
            replace(c, name=f'{variant}_{c.name}',
                    relative_path=str(Path(c.relative_path).with_name(f'fiber_{variant}_{c.name}.ome.zarr')))
            for c in original.channels)) for variant in VARIANTS)

    def load_model(self, *, device):
        if self.config['training']['mixed_precision'] != 'off':
            raise ValueError('TTA experiment requires precision off')
        if self.config.get('inference', {}).get('allow_tf32', False) is not False:
            raise ValueError('TTA requires allow_tf32=False')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return super().load_model(device=device)

    @torch.no_grad()
    def run_tile_inference(self, model, tile, *, device):
        if tile.shape[0] != 1 or len(set(tile.shape[-3:])) != 1:
            raise ValueError('TTA experiment requires batch1 cubic tiles')
        profile = not getattr(self, '_tta_profiled', False)
        def stamp():
            if profile and device.type == 'cuda':
                torch.cuda.synchronize(device)
            return time.monotonic()
        started = stamp()
        forward_seconds = 0.
        predictions = []
        tile = tile.to(device=device, non_blocking=True)
        for transform in TRANSFORMS:
            before = stamp()
            raw = super().run_tile_inference(model, transform_input(tile, transform), device=device)
            forward_seconds += stamp() - before
            predictions.append(reframe(raw, transform))
        reframed = stamp()
        result = torch.empty((tile.shape[0], 14, *predictions[0].shape[-3:]),
                             dtype=torch.float32, device=device)
        # Bound temporary sort/stack storage while keeping all results on GPU.
        for z in range(0, result.shape[2], 8):
            values = torch.stack([p[:, :, z:z + 8] for p in predictions])
            result[:, :7, z:z + 8] = encode_aggregate(values.mean(dim=0))
            result[:, 7:, z:z + 8] = encode_aggregate(midpoint_median(values))
        if profile:
            finished = stamp()
            print(f'TTA first-tile device={device} forward={forward_seconds:.3f}s '
                  f'reframe={reframed-started-forward_seconds:.3f}s '
                  f'aggregate={finished-reframed:.3f}s raw_device={result.device}', flush=True)
            self._tta_profiled = True
        return result


def selected_manifest(manifest, variant):
    """Scalar-channel view only; all variants retain shared cache ownership."""
    from lasagna.inference_provenance import atomic_write
    if variant not in VARIANTS:
        raise ValueError(f'unknown TTA variant {variant}')
    manifest = Path(manifest)
    document = json.loads(manifest.read_text())
    groups = {}
    for name in ('presence', 'nx', 'ny'):
        key = f'{variant}_{name}'
        group = document['groups'][key]
        groups[name] = {**group, 'channels': [name]}
    document['groups'] = groups
    destination = manifest.with_name(f'inference.{variant}.lasagna.json')
    if not destination.exists():
        atomic_write(destination, document, durable=False)
    elif json.loads(destination.read_text()) != document:
        raise ValueError('TTA manifest changed')
    return destination
