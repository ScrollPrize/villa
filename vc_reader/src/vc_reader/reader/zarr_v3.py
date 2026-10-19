"""Zarr v3 metadata normalized for the shared C++ chunk reader.

Supports regular grids, bytes plus optional Blosc/Zstd compression, and indexed
shards with bytes/CRC32C indexes. Other codec pipelines fail explicitly.
"""

import math
import numpy as np


def normalize(metadata):
    if metadata.get('zarr_format') != 3 or metadata.get('node_type') != 'array':
        raise ValueError('expected a Zarr v3 array')
    if metadata.get('storage_transformers'):
        raise ValueError('Zarr v3 storage transformers are not supported')
    dtype_name = metadata['data_type']
    if dtype_name not in ('bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                          'float16', 'float32', 'float64'):
        raise ValueError(f'unsupported Zarr v3 data_type {dtype_name!r}')
    dtype = np.dtype(dtype_name)
    shape = metadata['shape']
    grid = metadata['chunk_grid']
    if grid['name'] != 'regular':
        raise ValueError('Zarr v3 requires a regular chunk grid')
    outer = grid['configuration']['chunk_shape']
    keys = metadata['chunk_key_encoding']
    if keys['name'] not in ('default', 'v2'):
        raise ValueError('unsupported Zarr v3 chunk key encoding')
    separator = keys.get('configuration', {}).get('separator', '/' if keys['name'] == 'default' else '.')
    if separator not in ('/', '.'):
        raise ValueError('invalid Zarr v3 key separator')
    codecs = metadata['codecs']
    chunks = outer
    ratios = []
    index_end, index_crc = True, False
    if len(codecs) == 1 and codecs[0]['name'] == 'sharding_indexed':
        shard = codecs[0]['configuration']
        chunks = shard['chunk_shape']
        if len(outer) != len(chunks) or any(type(c) is not int or c <= 0 or type(s) is not int or s <= 0 or s % c for s, c in zip(outer, chunks)):
            raise ValueError('Zarr v3 inner chunks must evenly divide the shard shape')
        ratios = [s//c for s, c in zip(outer, chunks)]
        indexes = shard['index_codecs']
        names = [c['name'] for c in indexes]
        if names not in (['bytes'], ['bytes', 'crc32c']) or indexes[0].get('configuration', {}).get('endian') != 'little':
            raise ValueError('Zarr v3 shard index requires little-endian bytes, optionally CRC32C')
        index_crc = names[-1] == 'crc32c'
        location = shard.get('index_location', 'end')
        if location not in ('start', 'end'):
            raise ValueError('invalid shard index_location')
        index_end = location == 'end'
        if math.prod(ratios)*16+4 > 64 << 20:
            raise ValueError('Zarr v3 shard index exceeds the 64 MiB bounded index limit')
        codecs = shard['codecs']
    if len(shape) == 0 or len(shape) != len(chunks) or any(type(n) is not int or n < 0 for n in shape) or any(type(c) is not int or c <= 0 for c in chunks):
        raise ValueError('invalid Zarr v3 array/chunk shape')
    names = [c['name'] for c in codecs]
    if names not in (['bytes'], ['bytes', 'blosc'], ['bytes', 'zstd']):
        raise ValueError(f'unsupported Zarr v3 chunk codec pipeline {names}')
    endian = codecs[0].get('configuration', {}).get('endian')
    if dtype.itemsize > 1:
        if endian not in ('little', 'big'):
            raise ValueError('multi-byte Zarr v3 samples require explicit bytes endian')
        dtype = dtype.newbyteorder('<' if endian == 'little' else '>')
    compressor = None if len(codecs) == 1 else dict(codecs[1].get('configuration', {}), id=names[-1])
    fill = metadata['fill_value']
    if fill in ('NaN', 'Infinity', '-Infinity'):
        fill = float(fill)
    return dict(zarr_format=2, shape=shape, chunks=chunks, dtype=dtype.str, fill_value=fill,
                compressor=compressor, filters=None, order='C', dimension_separator=separator,
                _vc_reader_v3=dict(metadata=metadata, prefix='c'+separator if keys['name'] == 'default' else '',
                                   shard_chunks=ratios, index_end=index_end, index_crc=index_crc))
