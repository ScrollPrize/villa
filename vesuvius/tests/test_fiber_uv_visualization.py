import xml.etree.ElementTree as ET
import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_visualization import segments,progression,render_layout


def fixture():
    d=dict(phase=np.array([.1,.2,.3]),z=np.array([1.,2.,3.]),family=np.array([0,1,0]),
        node_ds=np.array([[0.,-5.,-10.],[0.,10.,20.],[0.,5.,10.]]),
        source_id=np.array([42,43,44]),start=np.zeros(3),end=np.array([10.,20.,10.]))
    state=np.zeros((3,9));state[:,:3]=np.log([[.6,.3,.1],[.2,.6,.2],[.1,.1,.8]])
    return d,state,np.array([100.,200.]),dict(layout_mode='shared_anchor',broken_threshold=.5,render_height=800)


def test_signed_endpoints_and_confidence():
    d,s,w,c=fixture();ids,x,y,joint,conditional,p=segments(d,s,w,c)
    np.testing.assert_array_equal(ids,[0,1])
    np.testing.assert_allclose(x[:,1]-x[:,0],[-10.,0.])
    np.testing.assert_allclose(y[:,1]-y[:,0],[0.,20.])
    np.testing.assert_allclose(joint,[.6,.6]);np.testing.assert_allclose(conditional,[2/3,.75])


def test_preupdate_alignment_and_missing_final():
    h=[dict(step=i,sign_violations=3-i,sign_violation_percent=i,retained_arc_percent=90,
            **({'pre_update_energy':10+i} if i else {})) for i in range(3)]
    rows=progression(h,99,2)
    assert [x['global_expected_cost'] for x in rows]==[13,14,101]
    h[1]['step']=3
    with pytest.raises(ValueError):progression(h,99,2)


def test_svg_complete_and_histogram_counts(tmp_path):
    d,s,w,c=fixture();path=tmp_path/'layout.svg'
    result=render_layout(path,'test',d,s,w,c)
    root=ET.parse(path).getroot();ns={'s':'http://www.w3.org/2000/svg'}
    lines=root.findall('.//s:line[@class="fiber"]',ns)
    assert len(lines)==2
    assert all(float(line.attrib['opacity'])==.6 for line in lines)
    assert not root.findall('.//s:title',ns)
    assert sum(result['joint_counts'])==sum(result['conditional_counts'])==2
    import tifffile
    with tifffile.TiffFile(path.with_suffix('.tif')) as tif:
        assert tif.pages[0].compression.name=='LZW'
        assert tif.pages[0].shape==(800,result['raster_size'][0],3)
        assert tif.asarray().min()<255
    scale=result['pixels_per_base_voxel']
    assert abs(float(lines[0].attrib['x2'])-float(lines[0].attrib['x1']))==pytest.approx(10*scale,abs=.002)
    assert abs(float(lines[1].attrib['y2'])-float(lines[1].attrib['y1']))==pytest.approx(20*scale,abs=.002)
    np.testing.assert_array_equal(result['winding_boundaries'],[0.,100.,300.])
    boundaries=root.findall('.//s:line[@class="winding-boundary"]',ns)
    assert [int(b.attrib['data-boundary']) for b in boundaries]==[0,1]
    assert [float(b.attrib['data-u']) for b in boundaries]==[0.,100.]
    xmin,xmax,_,_=result['bounds']
    for b in boundaries:
        assert float(b.attrib['x1'])==pytest.approx(90+(float(b.attrib['data-u'])-xmin)*scale)


def test_saved_visualization_artifacts_when_present():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'global-fiber-sgd/data/uv_categorical_v1/visualization'
    if not (root/'results.json').exists():pytest.skip('Optional saved visualization artifacts absent')
    for direction in ('ccw','cw'):
        meta=json.loads((root/direction/'summary.json').read_text());c=meta['confidence']
        import tifffile
        with tifffile.TiffFile(root/direction/'layout.tif') as tif:
            assert tif.pages[0].shape==(1000,c['raster_size'][0],3)
            assert tif.pages[0].compression.name=='LZW'
        assert sum(c['joint_counts'])==sum(c['conditional_counts'])==c['retained_pieces']
        count=0;boundaries=[]
        for _,element in ET.iterparse(root/direction/'layout.svg',events=['end']):
            assert not element.tag.endswith('}title')
            if element.tag.endswith('}line') and element.attrib.get('class')=='fiber':
                count+=1
                assert 0<=float(element.attrib['opacity'])<=1
                assert all(np.isfinite(float(element.attrib[k])) for k in ('x1','x2','y1','y2'))
            if element.attrib.get('class')=='winding-boundary':
                boundaries.append(int(element.attrib['data-boundary']))
            element.clear()
        assert count==c['retained_pieces']
        assert boundaries==c['visible_boundary_indices']


def test_tiled_raster_preserves_order_and_cross_tile_geometry(tmp_path):
    from PIL import Image,ImageDraw
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_visualization import TiledRaster
    expected=Image.new('RGB',(90,70),'white');reference=ImageDraw.Draw(expected,'RGBA')
    tiled=TiledRaster(90,70,tile=32)
    for draw in (reference,tiled):
        draw.line((3,18,85,18),fill=(27,108,168,130),width=1)
        draw.line((34,2,34,67),fill=(208,105,22,190),width=1)
        draw.rectangle((28,29,68,44),fill=(64,121,173,255))
        draw.text((25,49),'cross tile',fill=(30,30,30,255))
    path=tmp_path/'tiled.tif';tiled.save(path)
    np.testing.assert_array_equal(tifffile.imread(path),np.asarray(expected))
