import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).parents[2] / 'scripts/collect_reviewed_fibers.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def fixture(tmp_path):
    originals, split = tmp_path/'all', tmp_path/'split'
    parent = dict(type='vc3d_fiber', tags=[], vc_open_data_coordinate_space='PHerc1299/123@L0',
                  hv_classification=dict(automatic_tag='V', manual_tag='', control_point_length=100))
    for root in (originals, split/'train'):
        (root/'PHerc1229').mkdir(parents=True)
        (root/'PHerc1229/fiber.json').write_text(json.dumps(parent))
    piece = dict(parent, tags=['reviewed', 'rework'],
                 hv_classification=dict(automatic_tag='H', manual_tag='', control_point_length=10))
    return originals, split, piece


def test_parent_override_and_no_mutation(tmp_path):
    originals, split, piece = fixture(tmp_path)
    scroll, result, provenance = collector.inherit(piece, '0001_fiber_run0.json', originals, split)
    assert scroll == 'PHerc1299'
    assert result['hv_classification']['manual_tag'] == 'V'
    assert result['hv_classification']['control_point_length'] == 10
    assert piece['hv_classification']['manual_tag'] == ''
    assert provenance['parent_hv_classification']['control_point_length'] == 100


def test_reject_missing_unknown_test_and_wrong_volume(tmp_path):
    originals, split, piece = fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        collector.inherit(piece, '0001_missing_run0.json', originals, split)
    piece['vc_open_data_coordinate_space'] = 'PHerc1299/999@L0'
    with pytest.raises(ValueError, match='coordinate space'):
        collector.inherit(piece, '0001_fiber_run0.json', originals, split)
    piece['vc_open_data_coordinate_space'] = 'PHerc1299/123@L0'
    path = originals/'PHerc1229/fiber.json'
    doc = json.loads(path.read_text())
    doc['hv_classification']['automatic_tag'] = 'unknown'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='unresolved'):
        collector.inherit(piece, '0001_fiber_run0.json', originals, split)
    doc['hv_classification']['manual_tag'] = 'H'
    path.write_text(json.dumps(doc))
    (split/'test/PHerc1229').mkdir(parents=True)
    (split/'train/PHerc1229/fiber.json').rename(split/'test/PHerc1229/fiber.json')
    with pytest.raises(ValueError, match='exclusively in training'):
        collector.inherit(piece, '0001_fiber_run0.json', originals, split)


def test_download_selection_atomic_publish_and_repeat(tmp_path):
    originals, split, piece = fixture(tmp_path)
    source = 'https://example.test/'
    remote = {source: b'<a href="../">up</a><a href="1299_refine/">data</a>',
              source+'1299_refine/': b'<a href="0001_fiber_run0.json">yes</a><a href="skip.json">no</a>',
              source+'1299_refine/0001_fiber_run0.json': json.dumps(piece).encode(),
              source+'1299_refine/skip.json': json.dumps(dict(piece, tags=[])).encode()}
    output = tmp_path/'out'
    with patch.object(collector, 'fetch', side_effect=remote.__getitem__):
        manifest = collector.collect(source, originals, split, output)
        with pytest.raises(ValueError, match='overwrite'):
            collector.collect(source, originals, split, output)
    assert manifest['counts'] == {'PHerc1299': 1}
    assert manifest['reviewed_with_rework'] == 1
    assert len(list((output/'train').rglob('*.json'))) == 1
    raw = output/'raw_reviewed/1299_refine/0001_fiber_run0.json'
    assert raw.read_bytes() == remote[source+'1299_refine/0001_fiber_run0.json']
    remote[source+'1299_refine/skip.json'] = b'invalid json'
    with patch.object(collector, 'fetch', side_effect=remote.__getitem__):
        with pytest.raises(ValueError):
            collector.collect(source, originals, split, tmp_path/'failed')
    assert not (tmp_path/'failed').exists()
