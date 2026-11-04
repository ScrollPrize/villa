"""Public parsing behavior shared by the command-line entrypoints."""

import pytest

from vesuvius.utils.cli import HyphenUnderscoreParser


@pytest.mark.parametrize("declared", ["--chunk-size", "--chunk_size"])
def test_aliases_preserve_values_defaults_and_help(declared):
    parser = HyphenUnderscoreParser(prog="demo")
    parser.add_argument(declared, type=int, default=8)
    alternate = "--chunk_size" if declared == "--chunk-size" else "--chunk-size"

    assert parser.parse_args([]).chunk_size == 8
    help_text = parser.format_help()
    assert declared in help_text
    assert alternate not in help_text
    # Formatting help must not remove the aliases from subsequent parsing.
    for flag in (declared, alternate):
        assert parser.parse_args([flag, "4"]).chunk_size == 4


def test_short_flags_and_positionals():
    parser = HyphenUnderscoreParser()
    parser.add_argument("input_path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(["input", "--quiet", "-v"])
    assert vars(args) == {"input_path": "input", "quiet": True, "verbose": True}


def test_explicit_aliases_preserve_custom_destination():
    parser = HyphenUnderscoreParser()
    parser.add_argument("--model-type", "--model_type", dest="model", default="auto")
    assert parser.parse_args([]).model == "auto"
    for flag in ("--model-type", "--model_type"):
        assert parser.parse_args([flag, "train_py"]).model == "train_py"


@pytest.mark.parametrize("first, second", [("--a_b", "--a-b"), ("--a-b", "--a_b")])
def test_explicit_declarations_keep_separate_values(first, second):
    parser = HyphenUnderscoreParser()
    parser.add_argument(first, dest="first")
    parser.add_argument(second, dest="second")
    args = parser.parse_args([first, "one", second, "two"])
    assert vars(args) == {"first": "one", "second": "two"}
