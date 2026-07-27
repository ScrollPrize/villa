"""Long options should answer to both --foo-bar and --foo_bar, everywhere.

The three stages of the documented pipeline did not agree with each other. blend_logits
takes --chunk_size and --num_workers; finalize_outputs took --chunk-size and --num-workers
for the same two parameters, so running the pipeline end to end meant switching convention
between stage 2 and stage 3. predict has the same split internally: --model_path,
--model_cache_dir, --input_dir and --patch_size use underscores while --model-type uses a
hyphen.

The alternate spelling is derived automatically rather than hand-written per flag, so a
flag added later needs no thought.
"""

from __future__ import annotations

import pytest

from vesuvius.utils.cli import HyphenUnderscoreParser


# --- the mechanism -----------------------------------------------------------------

def test_derives_underscore_from_hyphen():
    p = HyphenUnderscoreParser()
    p.add_argument("--chunk-size", dest="chunk_size")
    assert p.parse_args(["--chunk-size", "1"]).chunk_size == "1"
    assert p.parse_args(["--chunk_size", "1"]).chunk_size == "1"


def test_derives_hyphen_from_underscore():
    p = HyphenUnderscoreParser()
    p.add_argument("--num_workers", type=int)
    assert p.parse_args(["--num_workers", "4"]).num_workers == 4
    assert p.parse_args(["--num-workers", "4"]).num_workers == 4


def test_leaves_alone_what_has_no_separator():
    p = HyphenUnderscoreParser()
    p.add_argument("--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    assert p.parse_args(["--quiet"]).quiet is True
    assert p.parse_args(["-v"]).verbose is True


def test_explicit_both_spellings_does_not_collide():
    """A flag that already declares both must not try to register a third."""
    p = HyphenUnderscoreParser()
    p.add_argument("--model-type", "--model_type", dest="model_type", default="auto")
    assert p.parse_args(["--model-type", "x"]).model_type == "x"
    assert p.parse_args(["--model_type", "x"]).model_type == "x"


def test_does_not_steal_a_name_another_flag_owns():
    """If --a_b is already taken, adding --a-b must not blow up constructing the CLI."""
    p = HyphenUnderscoreParser()
    p.add_argument("--a_b", dest="first")
    p.add_argument("--a-b", dest="second")      # would collide if aliased blindly
    assert p.parse_args(["--a-b", "x"]).second == "x"


def test_derived_spellings_are_hidden_from_help():
    p = HyphenUnderscoreParser(prog="demo")
    p.add_argument("--chunk-size", dest="chunk_size")
    text = p.format_help()
    assert "--chunk-size" in text
    assert "--chunk_size" not in text          # the alias works but does not clutter


def test_dest_is_unchanged():
    """dest comes from the declared option, not the derived one."""
    p = HyphenUnderscoreParser()
    a = p.add_argument("--num-workers", type=int)
    assert a.dest == "num_workers"


# --- the real parsers --------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--chunk-size", "--chunk_size"])
def test_finalize_accepts_both_chunk_size_spellings(flag):
    from vesuvius.models.run.finalize_outputs import build_parser
    args = build_parser().parse_args(["in", "out", flag, "64,64,64"])
    assert args.chunk_size == "64,64,64"


@pytest.mark.parametrize("flag", ["--num-workers", "--num_workers"])
def test_finalize_accepts_both_num_workers_spellings(flag):
    from vesuvius.models.run.finalize_outputs import build_parser
    assert build_parser().parse_args(["in", "out", flag, "3"]).num_workers == 3


@pytest.mark.parametrize("flag", ["--model-type", "--model_type"])
def test_predict_accepts_both_model_type_spellings(flag):
    from vesuvius.models.run.inference import build_parser
    args = build_parser().parse_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o", flag, "train_py"]
    )
    assert args.model_type == "train_py"


@pytest.mark.parametrize("flag", ["--patch-size", "--patch_size"])
def test_predict_accepts_both_patch_size_spellings(flag):
    """A flag nobody hand-aliased - it works because the derivation is automatic."""
    from vesuvius.models.run.inference import build_parser
    args = build_parser().parse_args(
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o", flag, "192,192,192"]
    )
    assert args.patch_size == "192,192,192"


def test_defaults_unchanged():
    """Adding aliases must not disturb anything that already worked."""
    from vesuvius.models.run.finalize_outputs import build_parser
    args = build_parser().parse_args(["in", "out"])
    assert args.chunk_size is None and args.num_workers is None
    assert args.input_path == "in" and args.output_path == "out"
