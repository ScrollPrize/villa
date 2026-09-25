"""Reproducible per-trace random streams, independent of batching and global RNG."""
import hashlib

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.model import initial_residuals


def trace_generator(seed, position, heading):
    """Identify a directed physical seed without relying on its batch index.

    CPU generators give the same initial noise on CPU/CUDA and on regrouping.
    Numerical model outputs can still differ across devices or batch shapes.
    """
    digest = hashlib.blake2b(digest_size=8, person=b'fiber-flow-rng')
    digest.update(str(int(seed)).encode('ascii'))
    digest.update(np.asarray(position, dtype='<f8').tobytes())
    digest.update(np.asarray(heading, dtype='<f8').tobytes())
    key = int.from_bytes(digest.digest(), 'little') & ((1 << 63)-1)
    return torch.Generator(device='cpu').manual_seed(key)


def trace_noise(cfg, generators, device):
    return torch.cat([initial_residuals(cfg, 1, 'cpu', generator) for generator in generators]).to(device)
