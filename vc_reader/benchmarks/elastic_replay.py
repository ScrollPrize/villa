"""Inspect trusted Elastic replay fields for tests and visualization."""
import struct
import numpy as np


def displacement(plan):
    """Inspect the first geometric step in the trusted VCA1 replay record."""
    data=plan._native.dump(); offset=4
    if data[:4]!=b'VCA1':raise ValueError('unsupported replay format')
    def number():
        nonlocal offset
        value=struct.unpack_from('<Q',data,offset)[0]; offset+=8; return value
    def sequence(dtype):
        nonlocal offset
        count=number(); value=np.frombuffer(data,dtype=dtype,count=count,offset=offset).copy()
        offset+=count*np.dtype(dtype).itemsize; return value
    def text():
        nonlocal offset
        count=number(); offset+=count
    for _ in range(number()): text(); sequence('<u8'); number(); number()
    assert number()==1
    sequence('<u8'); assert number()==0
    shape=tuple(sequence('<u8').astype(int)); sequence('<f8'); sequence('<f8')
    field=sequence('<f8')
    return np.zeros((*shape,len(shape))) if not len(field) else field.reshape(*shape,len(shape))

