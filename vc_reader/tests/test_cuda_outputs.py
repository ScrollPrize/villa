"""Consumer-owned outputs and device storage lifetime across CUDA streams."""
import gc
import os
from pathlib import Path
import sys

import numpy as np
import pytest

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import cuda, _reader, augment
from vc_reader.reader import ZarrReader
from vc_reader.sheet_normal import sheet_normals
from test_source_storage import array
from test_sheet_normal import examples


def test_cpu_normal_output():
    patches = examples()[:1]
    expected = sheet_normals(patches)
    out = np.empty((1,4),np.float32)
    assert sheet_normals(patches, out=out) is out
    np.testing.assert_array_equal(out, expected)
    for bad in [np.empty((4,),np.float32), np.empty((1,4),np.int32), np.empty((1,8),np.float32)[:,::2]]:
        with pytest.raises(ValueError):
            sheet_normals(patches,out=bad)
    floats = patches.astype(np.float32)
    with pytest.raises(ValueError, match='overlap'):
        sheet_normals(floats,out=floats.reshape(-1)[:4].reshape(1,4))


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA build/device required')
    return torch


def test_reader_output_and_validation(tmp_path, torch_cuda):
    torch = torch_cuda
    values = np.arange(16**3,dtype=np.uint8).reshape(16,16,16)
    array(tmp_path/'volume',values)
    origins = np.array([[1,2,3],[2,3,4]],np.float32)
    axes = np.broadcast_to(np.eye(3,dtype=np.float32),(2,3,3))
    with ZarrReader.open(tmp_path/'volume') as cpu, cpu.cuda() as gpu:
        for order, dtype in [(0,torch.uint8),(1,torch.float32)]:
            expected = gpu.read_affine_batch((3,4,5),origins,axes,order=order,source='chunks').numpy()
            for source in ['chunks','cpu']:
                out = torch.empty((2,3,4,5),dtype=dtype,device='cuda')
                returned, valid = gpu.read_affine_batch((3,4,5),origins,axes,order=order,source=source,out=out,valid=True)
                assert returned is out
                np.testing.assert_array_equal(out.cpu(),expected)
                assert valid.numpy().all()
        with pytest.raises(ValueError, match='shape'):
            gpu.read_affine_batch((3,4,5),origins,axes,out=torch.empty((120,),device='cuda'))
        with pytest.raises(TypeError):
            gpu.read_affine_batch((3,4,5),origins,axes,out=torch.empty((2,3,4,5),device='cuda',dtype=torch.uint8))
        with pytest.raises((TypeError,RuntimeError)):
            gpu.read_affine_batch((3,4,5),origins,axes,out=torch.empty((2,3,4,10),device='cuda')[...,::2])
    np.testing.assert_array_equal(out.cpu(), expected)


def test_normal_output_other_stream(torch_cuda):
    torch = torch_cuda
    host = examples()[:2]
    expected = sheet_normals(host)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        patches = torch.as_tensor(host,device='cuda')
        out = torch.empty((2,4),device='cuda')
        stream.synchronize()
        _reader._set_stream(stream.cuda_stream)
        try:
            assert sheet_normals(patches,out=out) is out
            result = out.clone()
            with pytest.raises(ValueError, match='out'):
                sheet_normals(patches,out=torch.empty((2,4),device='cuda',dtype=torch.int32))
            floats = patches.float()
            stream.synchronize()
            with pytest.raises(ValueError, match='overlap'):
                sheet_normals(floats,out=floats.flatten()[:8].reshape(2,4))
        finally:
            stream.synchronize()
            _reader._set_stream(0)
    np.testing.assert_allclose(result.cpu(),expected,atol=2e-5)


@pytest.mark.parametrize('inplace',[False,True])
def test_exported_augmentation_views_keep_inputs_alive(torch_cuda,inplace):
    torch = torch_cuda
    host = np.arange(64,dtype=np.float32).reshape(1,4,4,4)
    source = cuda.asarray(host)
    pipeline = augment.Pipeline([augment.BrightnessAdd(shift=3)],backend='cuda')
    output = pipeline({'image':source},seed=1,inplace=inplace)['image'].reshape(4,4,4)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        shared = torch.from_dlpack(output)
        torch.cuda._sleep(10_000_000)
        result = shared+1
        del source,output,shared
        gc.collect()
    stream.synchronize()
    np.testing.assert_array_equal(result.cpu(),host[0]+4)


def test_internal_release_and_owned_output_do_not_drain_other_stream(torch_cuda):
    torch = torch_cuda
    patches = cuda.asarray(examples()[:1])
    out = torch.empty((1,4),device='cuda')
    sheet_normals(patches,out=out)  # Warm allocation pool and kernels.
    unused = cuda.asarray(np.zeros((1,4,4,4),np.float32)).reshape(4,4,4)
    other = torch.cuda.Stream()
    done = torch.cuda.Event()
    with torch.cuda.stream(other):
        torch.cuda._sleep(500_000_000)
        done.record()
    try:
        sheet_normals(patches,out=out)
        del unused
        gc.collect()
        assert not done.query(), 'Internal release or out= synchronized unrelated CUDA work'
    finally:
        other.synchronize()
