"""Follower array interface backed by the shared native VC3D volume cache."""
import numpy as np


class NativeArray:
    def __init__(self, volume, level):
        if not volume.has_scale_level(level):
            raise ValueError(f'missing requested CT scale {level}')
        self.volume, self.level = volume, level
        self.shape = tuple(volume.shape_at(level))
        self.chunks = tuple(volume.chunk_shape(level))
        self.dtype = np.dtype(volume.dtype)
        self.fill = volume.fill_value

    def read(self, start, size):
        return self.volume.read_zyx(tuple(map(int, start)), tuple(map(int, size)), self.level)

    def prefetch(self, start, size):
        self.volume.prefetch_zyx(tuple(map(int, start)), tuple(map(int, size)), self.level, False)

    def sample_nearest(self, zyx):
        # This optional endpoint sampler uses the native bbox reader too.
        coords = np.rint(zyx).astype(np.int64)
        flat = coords.reshape(-1, 3)
        if not len(flat):
            return np.empty(coords.shape[:-1], self.dtype)
        lo, hi = flat.min(0), flat.max(0)+1
        block = self.read(lo, hi-lo)
        local = flat-lo
        return block[tuple(local.T)].reshape(coords.shape[:-1])
