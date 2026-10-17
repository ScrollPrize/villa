"""The full benchmark stays exhaustive and its previews show real slices."""
import importlib
from pathlib import Path
import re

import numpy as np
import pytest


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    return importlib.import_module('augment_catalog'), importlib.import_module('augment_preview')


def test_benchmark_covers_every_native_transform(monkeypatch):
    pytest.importorskip('PIL')
    catalog,_=modules(monkeypatch)
    reference=np.zeros((1,8,8,8),np.float32)
    names=[c.name for c in catalog.catalog(8,reference,include_utilities=True)]
    native=(Path(__file__).resolve().parents[1]/'cpp/augment/pipeline.hpp').read_text()
    variant=native.split('using Op = std::variant<',1)[1].split('>;',1)[0]
    expected={v.strip().replace('std::shared_ptr<OneOf>','OneOf') for v in variant.split(',')}
    assert set(names)==expected
    assert len(names)==len(set(names))
    defaults={c.name for c in catalog.catalog(8,reference)}
    assert defaults==expected-catalog.UTILITY_TRANSFORMS
    assert 'Normalize' in defaults


@pytest.mark.parametrize('include_utilities',[False,True])
def test_html_only_refresh_filters_without_running_or_changing_results(tmp_path,monkeypatch,include_utilities):
    import json
    import sys
    pytest.importorskip('PIL')
    modules(monkeypatch)
    refresh=importlib.import_module('refresh_augment_previews')
    def unexpected(*args,**kwargs):
        pytest.fail('HTML-only refresh must not read volumes or execute transforms')
    monkeypatch.setattr(refresh,'ZarrReader',type('Reader',(),{'open':unexpected}))
    monkeypatch.setattr(refresh.A,'Pipeline',unexpected)
    report={'size':8,'results':[dict(name=name,parameters=name+'()',sample='image',timings={},preview=name+'.png')
                                for name in ('Normalize','OneHot','Gamma')]}
    path=tmp_path/'8.json';original=json.dumps(report);path.write_text(original)
    argv=['refresh_augment_previews.py',str(tmp_path),'--html-only']
    if include_utilities:argv.append('--include-utilities')
    monkeypatch.setattr(sys,'argv',argv)
    refresh.main()
    assert path.read_text()==original
    page=(tmp_path/'index.html').read_text();csv=(tmp_path/'timings.csv').read_text()
    assert 'data-name="normalize"' in page and ',Normalize,' in csv
    assert 'data-name="gamma"' in page and ',Gamma,' in csv
    assert ('data-name="onehot"' in page)==include_utilities
    assert (',OneHot,' in csv)==include_utilities


@pytest.mark.parametrize('transform',['Normalize','Elastic'])
def test_preview_refresh_skips_utility_execution(tmp_path,monkeypatch,transform):
    import json
    import sys
    from contextlib import contextmanager
    pytest.importorskip('PIL')
    modules(monkeypatch)
    refresh=importlib.import_module('refresh_augment_previews')
    class Reader:
        @staticmethod
        @contextmanager
        def open(*args,**kwargs):yield Reader()
        def read(self,*args):return np.ones((8,8,8),np.uint8)
    executed=[]
    def pipeline(transforms,**kwargs):
        executed.extend(t._constructor.__name__ for t in transforms)
        if transform=='Elastic':
            from bench_augment_catalog import describe
            assert describe(transforms[0])=='Elastic(sigma=3, magnitude=1)'
        return lambda source,seed:source
    monkeypatch.setattr(refresh,'ZarrReader',Reader)
    monkeypatch.setattr(refresh.A,'Pipeline',pipeline)
    monkeypatch.setattr(refresh,'preview',lambda *args:{'seed':123})
    report={'size':8,'array':'unused','start':[0]*3,'reference_start':[0]*3,'results':[
        dict(name=name,parameters='Elastic(sigma=3, magnitude=1)' if name=='Elastic' else name+'()',sample='image',preview=name+'.png',preview_selection={'seed':123})
        for name in (transform,'OneHot')]}
    (tmp_path/'8.json').write_text(json.dumps(report))
    monkeypatch.setattr(sys,'argv',['refresh_augment_previews.py',str(tmp_path)])
    refresh.main()
    assert executed==[transform]


def test_orthogonal_preview_panels_are_enlarged(tmp_path,monkeypatch):
    Image=pytest.importorskip('PIL.Image')
    catalog,render=modules(monkeypatch)
    data=np.ones((1,8,8,8),np.float32)
    case=catalog.Workload('Invert',[])
    output=tmp_path/'preview.png'
    render.preview(output,case,{'image':data},{'image':np.zeros_like(data)},8,'Invert()',123)
    image=np.asarray(Image.open(output))
    # Original panels fill their tiles even when the input is smaller than 360px.
    assert (image[300:500,100:300]==255).all()
    assert (image[750:900,100:300]==0).all()


def test_preview_slices_intersect_off_center_effect(monkeypatch):
    pytest.importorskip('PIL')
    _,render=modules(monkeypatch)
    original=np.ones((24,28,32),np.float32)
    augmented=original.copy()
    augmented[2:5,20:24,3:7]=0
    before,after,delta=render.effect_slices(original,augmented)
    assert before==after
    assert 2<=after[0]<5 and 20<=after[1]<24 and 3<=after[2]<7
    assert all(np.count_nonzero(np.take(delta,i,axis=a)) for a,i in enumerate(after))


def test_preview_identity_and_resized_grids(monkeypatch):
    pytest.importorskip('PIL')
    _,render=modules(monkeypatch)
    source=np.ones((24,28,32),np.float32)
    before,after,delta=render.effect_slices(source,source)
    assert before==after==[11,13,15]
    before,after,delta=render.effect_slices(source,np.zeros((12,14,16),np.float32))
    assert len(before)==len(after)==3
    assert delta.shape==(12,14,16)


def test_interior_selection_ignores_elastic_boundary_fill(monkeypatch):
    pytest.importorskip('PIL')
    _,render=modules(monkeypatch)
    original=np.ones((24,28,32),np.float32)
    augmented=original.copy()
    augmented[0]=0;augmented[:,-1]=0;augmented[:,:,0]=0
    augmented[11:14,15:18,19:22]=.8
    before,after,delta=render.effect_slices(original,augmented,interior_fraction=.25)
    assert before==after
    assert 11<=after[0]<14 and 15<=after[1]<18 and 19<=after[2]<22
    # A constant/identity interior still chooses its center, including singleton axes.
    for shape in ((1,1,1),(1,3,5),(24,28,32)):
        source=np.ones(shape,np.float32)
        before,after,_=render.effect_slices(source,source,interior_fraction=.25)
        assert before==after==[(n-1)//2 for n in shape]


def test_small_changed_regions_get_visible_locator_boxes(monkeypatch):
    pytest.importorskip('PIL')
    _,render=modules(monkeypatch)
    delta=np.zeros((100,120),np.float32)
    delta[7,95]=1
    assert render.changed_bounds(delta)==(92,4,99,11)
    assert render.changed_bounds(np.ones_like(delta)) is None
    assert render.changed_bounds(np.zeros_like(delta)) is None
