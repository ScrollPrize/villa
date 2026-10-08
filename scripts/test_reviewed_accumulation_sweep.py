import copy
import json
import numpy as np
import numcodecs
import pytest
from reviewed_accumulation_sweep import variant, check_complete
from report_accumulation_repeats import compare_source, inspect_cache, histogram_stats


def test_precision_and_byte_budget_only():
    base = dict(output_root='original', evaluation=dict(gpu_manager=dict(label='original',devices='0,1,2,3,4,5'),
        inference=dict(chunk_size=64,accumulator_dtype='float16',emit_embeddings=False,
            accumulator_hot_slots=114688,accumulator_spill_segment_slots=2048,accumulator_workers=128,
            tile_size=256,max_batch_size=4,precision='bf16')))
    original = copy.deepcopy(base)
    a,b = [variant(base,dtype,1) for dtype in ('float16','float32')]
    assert base == original
    sa,sb = [c['evaluation']['inference'] for c in (a,b)]
    for k in ('accumulator_hot_slots','accumulator_spill_segment_slots'):
        assert sa[k]*18 == sb[k]*32
    for k in ('accumulator_workers','tile_size','chunk_size','max_batch_size','precision'):
        assert sa[k] == sb[k]
    assert a['evaluation']['inference_cache_namespace'] != b['evaluation']['inference_cache_namespace']
    assert variant(base,'float32',2)['output_root'] != b['output_root']


def test_cache_compare_counts_zeros_and_missing(tmp_path):
    codec=numcodecs.Blosc(cname='zstd',clevel=1)
    caches=[]
    for i in range(6):
        root=tmp_path/str(i)
        for channel in ('presence','nx','ny'):
            path=root/f'fiber_{channel}.ome.zarr/1'
            path.mkdir(parents=True)
            (path/'.zarray').write_text(json.dumps(dict(dtype='|u1',order='C',filters=None,
                dimension_separator='/',chunks=[64]*3,shape=[65,63,62],compressor=codec.get_config())))
            for z in range(2 if i==0 else 1):
                file=path/str(z)/'0'/'0'
                file.parent.mkdir(parents=True)
                a=np.zeros((64,64,64),np.uint8)
                a[0,0,0]=i
                file.write_bytes(codec.encode(a))
        caches.append(inspect_cache(root))
    result=compare_source(caches)
    row=result['presence'][0]
    assert row['voxels']==64*63*62
    assert row['common_chunks']==1 and row['union_chunks']==2
    assert row['histogram'][0]==64*63*62-1 and row['histogram'][1]==1
    assert row['max']==1 and row['p99']==0
    assert row['changed_chunks']==1
    assert histogram_stats([100]+[0]*255)['changed_percent']==0


def test_incomplete_guard(tmp_path):
    assert not check_complete(tmp_path)
    full=tmp_path/'full'
    full.mkdir()
    (full/'summary.json').write_text('{"completed":true}')
    with pytest.raises(RuntimeError):
        check_complete(tmp_path)
    for name in ('inference_summary.json','pipeline_profile.json'):
        (full/name).write_text('{}')
    assert check_complete(tmp_path)
