import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_surface import relative_key,digest


def test_relative_keys_and_digest(tmp_path):
    assert str(relative_key('prefix/x.tif','prefix/'))=='x.tif'
    assert str(relative_key('prefix/sub/mask.tif','prefix/'))=='sub/mask.tif'
    for key in ('outside/x.tif','prefix/../x.tif','prefix//x.tif','prefix/./x.tif','prefix/'):
        with pytest.raises(ValueError):relative_key(key,'prefix/')
    path=tmp_path/'empty';path.touch()
    assert digest(path)=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
