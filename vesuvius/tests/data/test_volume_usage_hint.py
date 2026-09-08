"""The usage hint Volume prints on failure must itself be a working call (#1333).

The segment example omitted scroll_id, and Volume(type="segment", segment_id=...)
raises "Could not determine energy/resolution for scroll None", so the recovery
suggested at the moment of an error led straight to a second error.
"""

from __future__ import annotations

import re

import pytest

from vesuvius.data.volume import Volume


def test_failure_hint_segment_example_carries_scroll_id(capsys, monkeypatch):
    # Volume.__init__ probes the EC2 metadata endpoint before it validates
    # arguments; keep the test off the network.
    import vesuvius.data.volume as volume_module

    monkeypatch.setattr(volume_module, "is_aws_ec2_instance", lambda: False)
    with pytest.raises(Exception):
        Volume(type="segment", segment_id=20230827161847)  # the documented-but-broken call
    out = capsys.readouterr().out
    example = re.search(r"segment = Volume\((.*)\)", out)
    assert example is not None, out
    assert "scroll_id=" in example.group(1)
