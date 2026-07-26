"""Both spellings of the mixed-convention flags should parse to the same destination.

The three stages of the documented pipeline did not agree with each other. blend_logits
takes --chunk_size and --num_workers; finalize_outputs took --chunk-size and --num-workers
for the same two parameters, so running the pipeline end to end meant switching convention
between stage 2 and stage 3. predict has the same split internally: --model_path,
--model_cache_dir, --input_dir and --patch_size use underscores while --model-type uses a
hyphen.

These assert the aliases, and that the underscore forms still land on the same dest as
before, so nothing that already worked changes.
"""

from __future__ import annotations

import pytest


def _finalize_parser():
    from vesuvius.models.run.finalize_outputs import build_parser
    return build_parser()


@pytest.mark.parametrize("flag", ["--chunk-size", "--chunk_size"])
def test_finalize_accepts_both_chunk_size_spellings(flag):
    args = _finalize_parser().parse_args(["in", "out", flag, "64,64,64"])
    assert args.chunk_size == "64,64,64"


@pytest.mark.parametrize("flag", ["--num-workers", "--num_workers"])
def test_finalize_accepts_both_num_workers_spellings(flag):
    args = _finalize_parser().parse_args(["in", "out", flag, "3"])
    assert args.num_workers == 3


@pytest.mark.parametrize("flag", ["--model-type", "--model_type"])
def test_predict_accepts_both_model_type_spellings(flag):
    from vesuvius.models.run.inference import build_parser
    args = build_parser().parse_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o", flag, "train_py"]
    )
    assert args.model_type == "train_py"


def test_finalize_defaults_unchanged():
    """Adding aliases must not disturb anything that already worked."""
    args = _finalize_parser().parse_args(["in", "out"])
    assert args.chunk_size is None
    assert args.num_workers is None
    assert args.input_path == "in"
    assert args.output_path == "out"


def test_predict_model_type_default_unchanged():
    from vesuvius.models.run.inference import build_parser
    args = build_parser().parse_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o"]
    )
    assert args.model_type == "auto"
