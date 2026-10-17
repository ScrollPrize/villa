"""The public README has portable, reproducible numerical previews and timings."""
import ast
import importlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    pytest.importorskip('PIL')
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    return importlib.import_module('build_augmentation_readme')


def test_axis_exports_keep_original_and_augmented_panels(builder,tmp_path):
    from PIL import Image
    catalog=importlib.import_module('augment_catalog')
    render=importlib.import_module('augment_preview')
    data=np.ones((1,8,8,8),np.float32)
    path=tmp_path/'combined.png'
    render.preview(path,catalog.Workload('Invert',[]),{'image':data},{'image':np.zeros_like(data)},128,'Invert()',123)
    exported=render.export_axis_previews(path,tmp_path/'axes','Invert')
    assert [p.name for p in exported]==['Invert-z.webp','Invert-y.webp','Invert-x.webp']
    with Image.open(path) as combined:
        for axis,path in enumerate(exported):
            left=render.PREVIEW_GAP+axis*(render.PREVIEW_TILE+render.PREVIEW_GAP)
            expected=combined.crop((left,render.PREVIEW_TOP,left+render.PREVIEW_TILE,render.PREVIEW_TOP+2*render.PREVIEW_ROW_HEIGHT))
            np.testing.assert_array_equal(np.asarray(Image.open(path)),np.asarray(expected))
    for path in exported:
        pixels=np.asarray(Image.open(path))
        assert pixels.shape==(844,360,3)
        assert (pixels[80:300,60:300]==255).all()
        assert (pixels[502:722,60:300]==0).all()
    bad=tmp_path/'wrong-layout.png';Image.new('RGB',(100,100)).save(bad)
    with pytest.raises(ValueError,match='three-axis preview'):
        render.export_axis_previews(bad,tmp_path/'axes','Wrong')


def test_named_parameters_and_speedups(builder):
    from vc_reader import augment as A
    names=builder.parameter_names()
    assert builder.named_recipe('BrightnessGradient(12, loc=0.5, strength=0.3)',names)=='A.BrightnessGradient(scale=12, loc=0.5, strength=0.3)'
    assert builder.named_recipe('FDA(reference_volume(shape=(1, 128, 128, 128), dtype=float32), blend=1, fraction=0.1)',names)=='A.FDA(reference=reference, blend=1, fraction=0.1)'
    row={'timings':{m:{'p50_ms':v} for m,v in [('cpu',40),('cuda_host',10),('cuda_resident',2)]}}
    assert builder.timing_table(row)[-1]=='| 40.000 ms | 10.000 ms | 2.000 ms | 4.0× | 20.0× |'
    report=json.loads((ROOT/'docs/augmentations/benchmark-128.json').read_text())
    reference=np.zeros((1,8,8,8),np.float32)
    for row in report['results']:
        call=ast.parse(row['recipe'],mode='eval').body
        kwargs={kw.arg:reference if isinstance(kw.value,ast.Name) and kw.value.id=='reference'
                else ast.literal_eval(kw.value) for kw in call.keywords}
        transform=getattr(A,call.func.attr)(**kwargs)
        assert transform._constructor.__name__==row['name']
        assert row['run_id'] in report['runs']


def test_readme_rebuilds_from_portable_snapshot(builder,tmp_path):
    snapshot=ROOT/'docs/augmentations/benchmark-128.json'
    output=tmp_path/'README.md';output.write_text('old README content')
    assets=tmp_path/'docs/augmentations'
    builder.build(snapshot,output,assets)
    text=output.read_text()
    assert text==(ROOT/'README.md').read_text()
    assert json.loads((assets/'benchmark-128.json').read_text())==json.loads(snapshot.read_text())
    assert 'old README content' not in text
    assert text.count('<summary><strong>Normalize</strong></summary>')==1
    assert '<summary><strong>OneHot</strong></summary>' not in text
    assert text.count('<details>')==61
    assert text.count('</details>')==61
    assert '<details open' not in text
    assert text.count('| Z slice · YX plane |')==59
    assert '### 2D tracing companions' in text
    images=set(re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text))
    assert len(images)==180
    for image in images:
        assert (tmp_path/image).is_file()
        assert Path(image).suffix=='.webp'
    assert '[▶ Watch Elastic animation (WebM)](docs/augmentations/animations/Elastic.webm)' in text
    assert (assets/'animations/Elastic.webm').read_bytes()==(ROOT/'docs/augmentations/animations/Elastic.webm').read_bytes()
    for recipe in re.findall(r'```python\n(.*?)\n```',text,re.S):ast.parse(recipe)
    # Snapshot-only regeneration is stable, including timing and slice metadata.
    builder.build(assets/'benchmark-128.json',output,assets)
    assert output.read_text()==text
