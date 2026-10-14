import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vesuvius.tifxyz_label_transfer.core import Surface
from fiber_surface_render import canvas, band_geometry, fiber_overlay, inspection_crop
from fiber_surface_render import DEFAULT_PIXEL_SIZE, DEFAULT_LINE_WIDTH
from fiber_surface_render import fold_pixels, write_fold_maps


def test_half_resolution_defaults_preserve_registration():
    assert DEFAULT_PIXEL_SIZE == 8 and DEFAULT_LINE_WIDTH == 4
    y, x = np.mgrid[:5, :7].astype(np.float32)
    s = Surface(x=x*100, y=y*100, z=x*0, scale_yx=(.01,.01))
    old_shape, old_factor = canvas(s, 4)
    shape, factor = canvas(s, DEFAULT_PIXEL_SIZE)
    assert shape == tuple((n-1)//2+1 for n in old_shape)
    old_xyz, _ = band_geometry(s, old_shape, old_factor, 0, old_shape[0])
    xyz, _ = band_geometry(s, shape, factor, 0, shape[0])
    np.testing.assert_allclose(xyz, old_xyz[::2, ::2])
    a = dict(piece=np.array([0,0]), source=np.array([1,1]), arc=np.array([0,1]),
             final_grid_uv=np.array([[1,2],[5,2]]))
    overlay, _ = fiber_overlay(a, shape, factor, DEFAULT_LINE_WIDTH)
    assert np.count_nonzero(overlay[:,25,3]) == 4


def test_registered_affine_surface_and_overlay():
    y, x = np.mgrid[:5, :7].astype(np.float32)
    s = Surface(x=100+x*100, y=200+y*100, z=300+x*20+y*30, scale_yx=(.01,.01))
    shape, factor = canvas(s, 25)
    assert shape == (17,25)
    xyz, valid = band_geometry(s, shape, factor, 0, shape[0])
    np.testing.assert_allclose(xyz[8,12], [400,400,420])
    assert valid.all()
    a = dict(piece=np.array([0,0,1,1]), source=np.array([4,4,4,4]),
             arc=np.array([0,1,4,5]), final_grid_uv=np.array([[1,2],[3,2],[5,2],[6,2]]))
    overlay, ids = fiber_overlay(a, shape, factor, 1)
    assert overlay.shape == (*shape,4)
    assert overlay[8,12,3] > 0
    assert overlay[8,16,3] == 0  # no line across separate pieces of one source
    assert len(ids) == 2


def test_anisotropic_grid_and_invalid_geometry():
    y, x = np.mgrid[:5, :7].astype(np.float32)
    valid = np.ones(x.shape, bool)
    valid[1,1] = False
    s = Surface(x=x*50, y=y*100, z=np.ones_like(x),
                scale_yx=(.01,.02), valid=valid)
    shape, factor = canvas(s, 25)
    assert shape == (17,13)
    xyz, mask = band_geometry(s, shape, factor, 0, shape[0])
    np.testing.assert_allclose(xyz[8,6], [150,200,1])
    assert not mask[4,2]


def test_four_times_thicker_lines():
    a = dict(piece=np.array([0,0]), source=np.array([4,4]), arc=np.array([0,1]),
             final_grid_uv=np.array([[5,10],[25,10]]))
    thin, _ = fiber_overlay(a, (30,30), np.ones(2), 2)
    thick, _ = fiber_overlay(a, (30,30), np.ones(2), 8)
    assert np.count_nonzero(thin[:,15,3]) == 2
    assert np.count_nonzero(thick[:,15,3]) == 8


def test_inspection_crop_preserves_attachment_graph():
    y,x=np.mgrid[:4,:8].astype(np.float32)
    s=Surface(x=x*100,y=y*100,z=x*0,scale_yx=(.01,.01))
    a=dict(final_grid_uv=np.array([[1.,1.],[5.,1.]]),piece=np.array([0,0]),
           source=np.array([7,7]),arc=np.array([0.,400.]),family=np.array([1,1]))
    cropped,b=inspection_crop(s,a,[2,6])
    shape,factor=canvas(cropped,100)
    xyz,_=band_geometry(cropped,shape,factor,0,shape[0])
    np.testing.assert_allclose(xyz[1,0],[200,100,0])
    np.testing.assert_array_equal(b['final_grid_uv'],[[-1,1],[3,1]])
    overlay,index=fiber_overlay(b,shape,factor,1)
    assert overlay[1,0,3]>0 and index[0]['family']=='V'
    np.testing.assert_array_equal(a['final_grid_uv'],[[1,1],[5,1]])


def test_fold_map_registration_and_crop(tmp_path):
    import json
    import tifffile
    from fiber_surface import geometry_stats
    # Concave final quad with opposing triangle normals, one valid normal quad.
    x=np.array([[0,100,200],[0,100,50]],dtype=np.float32)
    y=np.array([[0,0,0],[100,100,50]],dtype=np.float32)
    s=Surface(x=x,y=y,z=x*0+100,scale_yx=(.01,.01))
    shape,factors=canvas(s,25)
    pixels,folds,degenerate=fold_pixels(s,shape,factors)
    assert folds.tolist()==[[False,True]]
    assert not pixels[:,:4].any() and pixels[:,4:].all()
    assert geometry_stats(np.stack([x,y,x*0],-1))['opposing_triangle_quads']==folds.sum()
    cropped,_=inspection_crop(s,{},[1,3])
    sh,fa=canvas(cropped,25)
    info=write_fold_maps(tmp_path,cropped,sh,fa,np.full(sh,100,np.uint8),np.full(sh,255,np.uint8),1)
    assert info['folded_quads']==1
    assert json.loads((tmp_path/'folds.json').read_text())['source_quad_row_col']==[[0,1]]
    assert (tifffile.imread(tmp_path/'folded_quads.tif')==255).all()
    rgb=tifffile.imread(tmp_path/'ct_folds.tif')
    assert rgb.shape==(*sh,3) and (rgb[...,0]>rgb[...,1]).all()
    cropped.valid[0,0]=False
    assert not fold_pixels(cropped,sh,fa)[0].any()
