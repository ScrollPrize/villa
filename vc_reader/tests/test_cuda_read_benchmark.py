"""Planning regressions and opt-in CUDA coverage for the read experiments."""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vc_reader._native")
from vc_reader.reader import ZarrReader


@pytest.fixture
def benchmark(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"benchmarks"))
    bench=importlib.import_module("bench_cuda_reads")
    bench.write_validation_array(tmp_path)
    return bench,tmp_path


@pytest.mark.parametrize("dtype",["uint8","float32"])
def test_box_union_and_cpu_batch(benchmark,dtype):
    bench,path=benchmark
    grids=[(np.array([3,4,5]),np.array([-1,1,2]),np.eye(3)),
           (np.array([3,4,5]),np.array([-1,1,2]),np.eye(3)),
           (np.array([2,3,4]),np.array([8,9,10]),np.eye(3)),
           (np.array([0,2,3]),np.zeros(3,dtype=int),np.eye(3))]
    case=dict(kind="box",dtype=dtype,grids=grids)
    with ZarrReader.open(str(path)) as reader:
        start,shape,mapping,read,nchunks,local=bench.prepare(reader,case)
        unique=set()
        for sh,origin,_ in grids: unique.update(reader.chunk_indices(origin,sh))
        assert nchunks==len(unique)
        assert np.count_nonzero(mapping>=0)==len(unique)
        source=read()
        for idx in unique:
            slot=mapping[tuple(np.array(idx)-start//reader.chunks)]
            np.testing.assert_array_equal(source[slot],reader.read(np.array(idx)*reader.chunks,reader.chunks))
        expected=np.concatenate([reader.read(origin,sh,dtype=dtype).ravel() for sh,origin,_ in grids])
        np.testing.assert_array_equal(bench.cpu(reader,case,start,local),expected)


def test_integer_plan_filters_extreme_indices(benchmark):
    bench,path=benchmark
    pts=np.array([[0,0,0],[0,0,0],[8,9,10],[-2**63,1,1],[2**63-1,0,0]],np.int64)
    with ZarrReader.open(str(path)) as reader:
        case=dict(kind="integer",dtype="uint8",points=pts)
        start,shape,mapping,read,nchunks,local=bench.prepare(reader,case)
        assert nchunks==2
        np.testing.assert_array_equal(bench.cpu(reader,case,start,local),reader.read_points(pts))


def test_cuda_read_contract(benchmark):
    bench,_=benchmark
    binary=os.environ.get("VC_READER_CUDA_SAMPLE_LIB")
    if not binary:
        pytest.skip("set VC_READER_CUDA_SAMPLE_LIB to the optional experiment library")
    bench.validate(binary,True)
    bench.validate_reads(binary)
