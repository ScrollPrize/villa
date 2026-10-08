import numpy as np
from reviewed_width_paths import prior_means, final_excursion, variation


def test_prior_excludes_current_and_missing_prefix_propagates():
    np.testing.assert_allclose(prior_means([10,20,60]),[10,10,15])
    np.testing.assert_allclose(prior_means([10,np.nan,60]),[10,10,np.nan],equal_nan=True)
    np.testing.assert_allclose(prior_means([60,20,10]),[60,60,40])
    assert prior_means([]).size==0


def test_final_excursion_uses_last_inside_including_boundary():
    assert final_excursion([0,2,1,1.1,2],4)==3
    assert final_excursion([2,3],1)==0
    assert variation([10,10,np.nan])['cv_percent']==0


def test_native_scalar_query(tmp_path):
    import json
    import subprocess
    from reviewed_width_paths import BINARY
    from lasagna.tiled_predict3d import _create_omezarr
    groups={}
    for channel,dtype,value in [('presence',np.uint8,255),('width',np.float32,10),('nx',np.uint8,128),
                               ('embedding_000',np.float32,-2),('embedding_001',np.float32,1)]:
        a=_create_omezarr(str(tmp_path/channel),(4,4,4),0,1,2,channel,dtype=dtype)['0']
        a[:]=value
        if channel=='width':
            a[:,:,1]=14
            a[2:,2:,2:]=-1  # Nonzero invalid width keeps a physical chunk on disk.
        groups[channel]=dict(zarr=f'{channel}/0',scaledown=0,channels=[channel])
    (tmp_path/'nx/0/1/0/0').unlink()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(version=2,source_to_base=1,groups=groups)))
    points=tmp_path/'points.json'
    points.write_text(json.dumps([[.5,.5,.5],[2.5,2.5,2.5],[.5,.5,2.5],[-1,0,0]]))
    result=subprocess.run([str(BINARY),'sample-width',str(manifest),str(points)],capture_output=True,text=True,check=True,timeout=30)
    rows=json.loads(result.stdout)
    assert [r['status'] for r in rows]==['ok','invalid_width','missing_coverage','missing_coverage']
    assert rows[0]['width']==12 and rows[0]['presence']==1
    result=subprocess.run([str(BINARY),'sample-embeddings',str(manifest),str(points)],capture_output=True,text=True,check=True,timeout=30)
    np.testing.assert_allclose(json.loads(result.stdout)[0]['embedding'],[-2/np.sqrt(5),1/np.sqrt(5)],atol=1e-6)
