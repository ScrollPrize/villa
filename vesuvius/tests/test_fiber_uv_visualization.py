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
    c['_annotations']=[dict(uv=[0.,0.],text='123 D=201')]
    c['_background_geometry']=segments(d,s,w,c)
    c['_context_connections']=[([0.,0.],[20.,10.])]
    c['_bounds']=np.array([0.,299.,0.,30.],dtype=np.float32)
    c['_dotted']=[True,False]
    c['_red_underlay_indices']=[0]
    result=render_layout(path,'test',d,s,w,c)
    import json
    json.dumps(result)  # Supplied float32 geometry bounds must serialize too.
    root=ET.parse(path).getroot();ns={'s':'http://www.w3.org/2000/svg'}
    lines=root.findall('.//s:line[@class="fiber"]',ns)
    assert len(lines)==2
    assert lines[0].attrib['stroke-dasharray']=='1 4'
    assert 'stroke-dasharray' not in lines[1].attrib
    assert [line.attrib['stroke'] for line in lines]==['#1b6ca8','#d06916']
    background=root.find('s:g[@id="individual-probability-background"]',ns)
    connections=root.find('s:g[@id="constraint-connections"]',ns)
    foreground=root.find('s:g[@id="retained-fiber-pieces"]',ns)
    underlays=root.find('s:g[@id="sign-violation-underlays"]',ns)
    assert len(underlays)==1
    assert underlays[0].attrib['stroke']=='#f9d2d2'
    assert float(underlays[0].attrib['stroke-width'])==1.3
    assert list(root).index(underlays)<list(root).index(foreground)
    assert len(background)==2
    assert list(root).index(background)<list(root).index(connections)<list(root).index(foreground)
    assert all(x.attrib['stroke']=='#cbdfee' for x in background)
    context=root.find('s:g[@id="context-connections"]',ns)
    assert list(root).index(background)<list(root).index(context)<list(root).index(connections)
    assert context[0].attrib['stroke-dasharray']=='6 4'
    assert all(float(line.attrib['opacity'])==.6 for line in lines)
    assert not root.findall('.//s:title',ns)
    assert '123 D=201' in path.read_text()
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


def test_dashed_segments():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_visualization import dashed_segments
    assert list(dashed_segments((0,0),(15,0)))==[(0,0,6,0),(10,0,15,0)]
    assert list(dashed_segments((0,0),(0,0)))==[]


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


def test_map_only_and_vector_pdf(tmp_path):
    import tifffile
    d,s,w,c=fixture();c.update(render_height=2000,raster_map_only=True,vector_pdf=True)
    result=render_layout(tmp_path/'layout.svg','test',d,s,w,c)
    assert result['raster_map_only']
    xmin,xmax,ymin,ymax=result['bounds']
    scale=result['raster_pixels_per_base_voxel']
    assert scale==pytest.approx(1999/(ymax-ymin))
    assert result['raster_size']==[int(np.ceil((xmax-xmin)*scale))+1,2000]
    with tifffile.TiffFile(tmp_path/'layout.tif') as tif:
        assert tif.pages[0].shape==(2000,result['raster_size'][0],3)
        assert tif.pages[0].compression.name=='LZW'
    assert tifffile.imread(tmp_path/'histograms.tif').shape==(350,1600,3)
    ET.parse(tmp_path/'histograms.svg')
    pdf=(tmp_path/'layout.pdf').read_bytes()
    assert pdf.startswith(b'%PDF-')
    assert b'/Subtype /Image' not in pdf
    assert b'/Type /Page' in pdf


def test_fresh_layout_only_without_gpu(tmp_path,monkeypatch):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_visualization as vis
    d,s,w,c=fixture();root=tmp_path/'input';root.mkdir()
    cache=root/'tensor.npz';np.savez(cache,**d)
    c['tensor_cache']=str(cache)
    (root/'summary.json').write_text(json.dumps({'config':c}))
    np.savez(root/'solution.npz',state=s,widths=w,**{k:d[k] for k in ('source_id','start','end','family')})
    def forbidden(*args,**kwargs):raise AssertionError('Layout must not request GPU')
    monkeypatch.setattr(vis.uv,'launch_tensor_solve',forbidden)
    reports=vis.run(dict(output_root=str(tmp_path/'output'),layout_only=True,
        render_height=900,raster_map_only=True,solutions=[dict(label='test',run_root=str(root))]))
    assert reports[0]['confidence']['raster_size'][1]==900
    assert reports[0]['confidence']['retained_pieces']==2
    np.savez(root/'solution.npz',state=s,widths=w,**dict({k:d[k] for k in ('start','end','family')},source_id=d['source_id']+1))
    with pytest.raises(ValueError,match='identity mismatch'):
        vis.run(dict(output_root=str(tmp_path/'bad'),layout_only=True,solutions=[dict(label='test',run_root=str(root))]))
