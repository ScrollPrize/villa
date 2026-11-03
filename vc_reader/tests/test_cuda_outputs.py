"""Consumer-owned outputs and device storage lifetime across CUDA streams."""
import gc

import numpy as np
import pytest

from build_support import load_build
load_build()

from vc_reader import cuda, transforms
from vc_reader.reader import ZarrReader
from vc_reader.sheet_normal import sheet_normals
from test_source_storage import array
from test_sheet_normal import examples


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


@pytest.mark.parametrize('inplace',[False,True])
def test_exported_augmentation_views_keep_inputs_alive(torch_cuda,inplace):
    torch = torch_cuda
    host = np.arange(64,dtype=np.float32).reshape(1,4,4,4)
    source = cuda.asarray(host)
    pipeline = transforms.Pipeline([transforms.BrightnessAdd(shift=3)],backend='cuda')
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
