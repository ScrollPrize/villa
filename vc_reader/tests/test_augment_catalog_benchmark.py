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
    names=[c.name for c in catalog.catalog(8,reference)]
    native=(Path(__file__).resolve().parents[1]/'cpp/augment/pipeline.hpp').read_text()
    variant=native.split('using Op = std::variant<',1)[1].split('>;',1)[0]
    expected={v.strip().replace('std::shared_ptr<OneOf>','OneOf') for v in variant.split(',')}
    assert set(names)==expected
    assert len(names)==len(set(names))


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
