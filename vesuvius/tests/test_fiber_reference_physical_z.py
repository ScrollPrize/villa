from types import SimpleNamespace
import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import physical_z_column_mask,projected_surface_z


def test_affine_vertical_alignment_preserves_relative_v():
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import fit_vertical_affine
    v=np.array([0.,2.,5.,10.])
    for slope in (2.,-2.):
        a,b=fit_vertical_affine(v,slope*v+17)
        np.testing.assert_allclose([a,b],[slope,17])
        np.testing.assert_allclose(np.diff(a*v+b),slope*np.diff(v))
    with pytest.raises(ValueError,match='Degenerate'):
        fit_vertical_affine(np.ones(3),np.arange(3))


def test_affine_reference_keeps_original_uv_geometry(tmp_path):
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import render_reference
    # Nonmonotonic physical Z must not reorder or collapse the original UV rows.
    z=np.array([[90.,90.],[10.,10.],[50.,50.]])
    x=np.ones_like(z);x[1,0]=-1
    surface=SimpleNamespace(full=[x,np.ones_like(z),z],
        meta={'output_step_vx':10},x=np.array([0.,1.]),q=np.array([1.,2.]))
    for slope,offset in ((1.,20.),(-1.,40.)):
        render_reference(surface,tmp_path,61,[0,60],1.,[slope,offset])
        image=tifffile.imread(tmp_path/'reference_own.tif')
        for row in (0,2):
            mapped=int(slope*row*10+offset)
            assert np.any(image[mapped,0]!=255)
        assert np.all(image[30,0]==255)
        assert np.any(image[30,-1]!=255)


def test_shared_mapping_endpoints_and_reversed_rows():
    z=np.array([[0.,80000.],[80000.,0.]])
    mask=physical_z_column_mask(z,np.ones_like(z,dtype=bool),[0,80000],1000)
    assert mask.shape==(1000,2)
    assert mask.all()


def test_does_not_bridge_missing_reference_rows():
    z=np.array([[0.],[10.],[20.],[30.]])
    valid=np.array([[True],[True],[False],[True]])
    mask=physical_z_column_mask(z,valid,[0,30],31)
    assert mask[:11].all()
    assert not mask[11:].any()


def test_projection_z_matches_triangle_not_bilinear():
    xyz=np.zeros((2,2,3));xyz[1,1,2]=100
    surface=SimpleNamespace(xyz=xyz,stride=2)
    z=projected_surface_z(surface,np.array([[.5,1.5],[1.5,.5],[2,2],[np.nan,np.nan]]))
    np.testing.assert_allclose(z[:3],[25,25,100])
    assert np.isnan(z[3])


def test_reference_tiff_preserves_empty_physical_z_ends(tmp_path):
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import render_reference
    z=np.array([[20.,20.],[80.,80.]])
    surface=SimpleNamespace(full=[np.ones_like(z),np.ones_like(z),z],
        meta={'output_step_vx':10},x=np.array([0.,1.]),q=np.array([1.,2.]))
    render_reference(surface,tmp_path,101,[0,100],1.)
    with tifffile.TiffFile(tmp_path/'reference_own.tif') as src:
        assert src.pages[0].compression.name=='LZW'
        image=src.asarray()
    assert image.shape==(101,11,3)
    assert np.all(image[:20]==255) and np.all(image[81:]==255)
    assert np.all(np.any(image[20:81]!=255,axis=2))
    assert not (tmp_path/'reference_z_raster.tmp').exists()
