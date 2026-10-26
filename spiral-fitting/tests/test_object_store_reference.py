"""The file loader still reproduces the captured object-store reference.

tests/data/object_store_reference.json is the comparison target for the
object-store adapter; this test pins it to the file loader that produced it
until that loader is removed.
"""
import pytest

import object_store_reference as reference


def assert_matches_reference(summary):
    expected = reference.load_reference()
    assert summary['constraints'] == expected['constraints']
    for name, values in expected['losses'].items():
        assert summary['losses'][name] == pytest.approx(values, abs=2e-5), name


def test_file_loader_reproduces_the_reference(tmp_path, monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    context = reference.file_context(str(tmp_path))
    assert_matches_reference(reference.reference_summary(context))
