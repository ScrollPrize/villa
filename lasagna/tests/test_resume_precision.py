import json

import pytest

from lasagna.inference_provenance import check_resume_precision


def _write(path, inference):
    path.write_text(json.dumps({"inference": inference}))


def test_new_output_is_not_checked(tmp_path):
    check_resume_precision(tmp_path / "inference.json", "fp32")


@pytest.mark.parametrize("recorded,current", [("bf16", "bf16"), ("off", "fp32"), ("fp32", "off")])
def test_same_precision_resumes(tmp_path, recorded, current):
    path = tmp_path / "inference.json"
    _write(path, {"precision": recorded})
    check_resume_precision(path, current)


@pytest.mark.parametrize("recorded,current", [("bf16", "off"), ("bf16", "fp32"), ("fp32", "bf16")])
def test_precision_change_refuses_to_mix(tmp_path, recorded, current):
    path = tmp_path / "inference.json"
    _write(path, {"precision": recorded})
    with pytest.raises(ValueError, match="would mix"):
        check_resume_precision(path, current)


def test_unrecorded_precision_warns_but_resumes(tmp_path, capsys):
    path = tmp_path / "inference.json"
    _write(path, {"tile_size": 192})
    check_resume_precision(path, "fp32")
    assert "does not record the inference precision" in capsys.readouterr().out
