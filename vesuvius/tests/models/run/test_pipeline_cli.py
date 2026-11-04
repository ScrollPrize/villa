"""CLI compatibility and validation at the model pipeline entrypoints."""

import importlib
import sys

import pytest


@pytest.fixture(params=["predict", "blend_logits", "blend_and_finalize", "finalize_outputs"])
def pipeline_cli(request):
    name = request.param
    module_name = {
        "predict": "inference",
        "blend_logits": "blending",
        "blend_and_finalize": "blending",
        "finalize_outputs": "finalize_outputs",
    }[name]
    module = importlib.import_module(f"vesuvius.models.run.{module_name}")
    factory = module.build_blend_and_finalize_parser if name == "blend_and_finalize" else module.build_parser
    required = (
        ["--model_path", "m", "--input_dir", "i", "--output_dir", "o"]
        if name == "predict" else ["in", "out"]
    )
    return name, factory(), required


@pytest.mark.parametrize("separator", ["-", "_"])
def test_pipeline_accepts_both_flag_spellings(pipeline_cli, separator):
    name, parser, required = pipeline_cli
    values = (
        {"model_type": "train_py", "patch_size": "192,192,192",
         "zarr_compressor": "lz4", "zarr_compression_level": 5}
        if name == "predict" else {"chunk_size": "64,64,64", "num_workers": 3}
    )
    if name in {"finalize_outputs", "blend_and_finalize"}:
        values["threshold_value"] = 0.3
        required = [*required, "--threshold"]
    if name == "finalize_outputs":
        required = [*required, "--delete" + separator + "intermediates"]
    argv = list(required)
    for dest, value in values.items():
        argv.extend(["--" + dest.replace("_", separator), str(value)])

    args = parser.parse_args(argv)

    for dest, value in values.items():
        assert getattr(args, dest) == value
    if name == "finalize_outputs":
        assert args.delete_intermediates is True


def test_finalize_defaults():
    from vesuvius.models.run.finalize_outputs import build_parser

    args = build_parser().parse_args(["in", "out"])
    assert (args.input_path, args.output_path) == ("in", "out")
    assert args.chunk_size is None and args.num_workers is None


@pytest.mark.parametrize("mode, threshold_args, expected", [
    ("binary", [], None),
    ("multiclass", ["--threshold"], 0.5),
    ("binary", ["--threshold", "--threshold-value", "0.3"], 0.3),
])
def test_finalize_main_dispatches_parsed_settings(monkeypatch, tmp_path, mode, threshold_args, expected):
    from vesuvius.models.run import finalize_outputs

    calls = []
    monkeypatch.setattr(finalize_outputs, "finalize_logits", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "vesuvius.finalize_outputs", str(tmp_path / "in.zarr"), str(tmp_path / "out.zarr"),
        "--mode", mode, *threshold_args,
    ])

    assert finalize_outputs.main() == 0
    assert len(calls) == 1
    assert calls[0]["mode"] == mode
    assert calls[0]["threshold"] == expected


@pytest.mark.parametrize("module_name, argv, message", [
    ("inference", ["--model_path", "m", "--input_dir", "i", "--output_dir", "o", "--bbox", "1,2,3"], "bbox"),
    ("finalize_outputs", ["in.zarr", "out.zarr", "--num_parts", "2", "--part_id", "5"], "part_id"),
])
def test_invalid_arguments_report_usage_error(monkeypatch, capsys, module_name, argv, message):
    module = importlib.import_module(f"vesuvius.models.run.{module_name}")
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err.lower()
