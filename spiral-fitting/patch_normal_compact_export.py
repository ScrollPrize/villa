"""Stream signed patch normals straight into the fitter's compact pool format."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

import numpy as np

from prepacked_patch_normals import pack_rows


ENCODING = {
    'name': 'signed_unit_vector_u8', 'components': ['nx', 'ny', 'nz'],
    'encode': 'round_ties_to_even(127 * clip(component, -1, 1)) + 128',
    'decode': '(float32(value) - 128) / 127; normalize the three-vector',
    'offset': 128, 'scale': 127, 'renormalize_after_decode': True,
    'component_zero': 128, 'component_min': 1, 'component_max': 255,
    'missing_vector': [0, 0, 0], 'presence': 'any component != 0',
}


class CompactPatchNormalWriter:
    """Write occupied 16³ bricks as bitmap/rank data while tiles are produced."""

    def __init__(self, output, shape, cell_size, base_scale, chunk_edge, metadata,
                 *, brick_edge=None):
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(self.output)
        if brick_edge is None:
            brick_edge = math.gcd(int(chunk_edge), 16)
        if (not np.isfinite(cell_size) or cell_size <= 0 or
                not np.isfinite(base_scale) or base_scale <= 0 or
                brick_edge < 1 or chunk_edge % brick_edge or brick_edge ** 3 > 32767):
            raise ValueError('invalid compact patch-normal grid or brick size')
        self.shape = tuple(map(int, shape))
        self.cell_size = float(cell_size)
        self.chunk_edge = int(chunk_edge)
        self.brick_edge = int(brick_edge)
        self.grid = tuple((s + brick_edge - 1) // brick_edge for s in self.shape)
        self.count = self.valid_count = 0
        self.rows = self.total_values = 1
        self.written = set()
        self.lock = threading.Lock()
        self.output.mkdir(parents=True)
        self.sidecar = self.output / 'signed_normals_u8.respool'
        self.sidecar.mkdir()
        self.table = np.lib.format.open_memmap(
            self.sidecar / 'table.npy', mode='w+', dtype=np.int32, shape=self.grid)
        self.table[:] = 0
        self.files = {name: (self.sidecar / name).open('wb') for name in
                      ('brick_coords.i32', 'bits.i64', 'prefix.i16', 'offsets.i64', 'values.u8')}
        self.files['brick_coords.i32'].write(np.asarray([[-1, -1, -1]], dtype=np.int32).tobytes())
        words = (brick_edge ** 3 + 63) // 64
        self.files['bits.i64'].write(bytes(words * 8))
        self.files['prefix.i16'].write(bytes(words * 2))
        self.files['offsets.i64'].write(np.asarray([1, 1], dtype=np.int64).tobytes())
        self.files['values.u8'].write(b'\0\0\0')
        self.info = dict(artifact_type='signed_patch_normal_volume', format_version=1,
                         complete=False, source_metadata=metadata,
                         shape_zyx=list(self.shape), cell_size_fitter_voxels=cell_size,
                         fitter_voxel_scale_base_voxels=base_scale,
                         scale_base_voxels=cell_size * base_scale,
                         translation_base_voxels=.5 * cell_size * base_scale,
                         chunks_zyx=[chunk_edge] * 3,
                         normal_encoding='signed unit vector uint8, vector order XYZ in compact pool',
                         channels={'normals': 'signed_normals_u8.respool'})
        self._write_manifest()

    def _write_manifest(self):
        (self.output / 'manifest.json').write_text(json.dumps(self.info, indent=2) + '\n')

    def reuse_chunk(self, _key):
        raise ValueError('compact export does not support resume')

    def append(self, result):
        valid = np.asarray(result['sign_valid'], dtype=bool)
        p = np.asarray(result['position_zyx'])[valid]
        n = np.asarray(result['normal_zyx'])[valid]
        if not len(p):
            with self.lock:
                self.count += len(valid)
            return
        if not np.isfinite(p).all() or not np.isfinite(n).all():
            raise ValueError('patch-normal positions and vectors must be finite')
        cells = np.floor(p / self.cell_size).astype(np.int64)
        if not ((cells >= 0) & (cells < self.shape)).all():
            raise ValueError('patch-normal cell outside output shape')
        chunk = cells // self.chunk_edge
        if not (chunk == chunk[0]).all():
            raise ValueError('each append must contain one output chunk')
        key = tuple(map(int, chunk[0]))
        # Detect collisions before writing anything for this tile.
        linear = np.ravel_multi_index(cells.T, self.shape)
        if len(np.unique(linear)) != len(linear):
            raise ValueError('tile contains duplicate output cells')
        encoded = np.rint(127 * np.clip(n[:, ::-1], -1, 1)).astype(np.int16) + 128
        encoded = encoded.astype(np.uint8)
        bricks = cells // self.brick_edge
        unique, inverse = np.unique(bricks, axis=0, return_inverse=True)
        edge = self.brick_edge
        local = cells % edge
        index = (local[:, 0] * edge + local[:, 1]) * edge + local[:, 2]
        values = np.zeros((len(unique), edge ** 3, 3), dtype=np.uint8)
        values[inverse, index] = encoded
        bits, prefix, counts, packed = pack_rows(values)
        with self.lock:
            if key in self.written:
                raise ValueError('output chunk written twice')
            self.written.add(key)
            self.count += len(valid)
            self.valid_count += len(p)
            coords = unique.astype(np.int32)
            self.table[tuple(coords.T)] = np.arange(self.rows, self.rows + len(coords), dtype=np.int32)
            self.files['brick_coords.i32'].write(coords.tobytes())
            self.files['bits.i64'].write(bits.tobytes())
            self.files['prefix.i16'].write(prefix.tobytes())
            self.files['values.u8'].write(packed.tobytes())
            offsets = self.total_values + counts.cumsum()
            self.files['offsets.i64'].write(offsets.tobytes())
            self.total_values = int(offsets[-1])
            self.rows += len(coords)

    def finish(self, output, metadata):
        if not self.valid_count:
            raise ValueError('no valid signed normals in the requested ROI')
        if self.total_values != self.valid_count + 1:
            raise ValueError('compact value count differs from occupied cell count')
        for stream in self.files.values():
            stream.close()
        self.table.flush()
        del self.table
        sidecar_meta = dict(format='compact_patch_normals', version=1,
                            array_shape=list(self.shape), brick_shape=[self.brick_edge] * 3,
                            grid_shape=list(self.grid), rows=self.rows,
                            total_values=self.total_values, normal_encoding=ENCODING,
                            source_metadata={'cell_size_fitter_voxels': self.cell_size})
        (self.sidecar / 'meta.json').write_text(json.dumps(sidecar_meta, indent=2) + '\n')
        self.info.update(source_metadata=metadata, input_rows=self.count,
                         invalid_sign_rows=self.count - self.valid_count,
                         occupied_cells=self.valid_count, spatial_chunks=len(self.written),
                         compact_bricks=self.rows - 1, complete=True)
        self._write_manifest()
