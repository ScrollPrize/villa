from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import lasagna.manager.cli as manager_cli
from lasagna.manager import catalog
from lasagna.manager.catalog import CatalogCache, get_catalog, index_volumes, resolve_volume
from lasagna.manager.cli import (
    COMMANDS, _completion_script, _expand_command, _render_volume_table,
    _resolve_token, _rewrite_contextual_help, main,
)
from lasagna.manager.completion import (
    canonical_executable, contextual_candidates, install_bash_completion, provider_id,
)
from lasagna.manager.config import ManagerConfig, initialize_config, load_config
from lasagna.manager.fiberlets import (
    finalize_fiberlet_provenance, launch_fiberlet, resolve_fiberlet_inputs,
    resume_fiberlet,
)
from lasagna.manager.prefetch import (
    build_prefetch_request, execute_prefetch_request, prefetch_volume,
    volume_cache_root,
)
from lasagna.manager.runner import main as runner_main
from lasagna.manager.runs import atomic_json, launch_inference, read_runs, reconcile_runs
from lasagna.manager.snapshots import (
    completion_snapshot_candidates, discover_snapshot_paths, index_snapshots,
    resolve_snapshot,
)
from lasagna.manager.tmux import Tmux


def configured(tmp_path: Path, *, snapshot_dirs=(), output=True, venv=True) -> ManagerConfig:
    return ManagerConfig(
        snapshot_dirs=tuple(str(path) for path in snapshot_dirs),
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs") if output else "",
        venv=str(tmp_path / "venv") if venv else "",
    )


def sample_catalog() -> dict:
    def volume(volume_id: str, long_id: str):
        return {
            "id": volume_id,
            "sample_id": "PHerc0001",
            "long_id": long_id,
            "properties": {
                "shape": [12, 10, 8],
                "pixel_size_um": 2.4,
                "data_format": "uint8",
                "license": {"name": "CC BY-NC 4.0", "url": "https://example/license"},
            },
            "data": [{
                "type": "ome-zarr",
                "origins": [{
                    "path": f"PHerc0001/volumes/{long_id}/",
                    "access_roots": [{"type": "s3", "url": "s3://public", "usage": "public-read"}],
                }],
            }],
        }
    return {
        "samples": {
            "PHerc0001": {
                "sample": {"id": "PHerc0001"},
                "volumes": {
                    "one": volume("20260101000001", "20260101000001-2.4um.zarr"),
                    "two": volume("20260101000002", "20260101000002-2.4um.zarr"),
                },
            }
        },
        "models": {},
    }


def test_config_init_round_trip_and_no_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.toml"
    monkeypatch.setenv("LAS_MANAGER_CONFIG", str(path))
    assert initialize_config() == path
    loaded = load_config()
    assert loaded.catalog_max_age_seconds == 3600
    assert loaded.snapshot_dirs == ()
    assert loaded.atlas_dir == ""
    assert loaded.upload_staging_s3 == ""
    assert loaded.fiberlet_binary == ""
    assert loaded.fiberlet_threads == 32
    assert loaded.fiberlet_params == ()
    assert loaded.params == (
        "--tile-size", "512", "--border", "32", "--overlap", "96",
        "--devices", "all",
    )
    assert loaded.rclone_params == (
        "--s3-provider", "AWS", "--s3-env-auth", "--transfers", "512",
        "--buffer-size", "2M", "--size-only", "--fast-list", "-P",
        "--stats-one-line",
    )
    with pytest.raises(FileExistsError):
        initialize_config()


def test_relative_config_paths_resolve_from_config_location(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.toml"
    path.parent.mkdir()
    path.write_text('cache_dir = "../cache"\nsnapshot_dirs = ["../runs"]\n', encoding="utf-8")
    monkeypatch.setenv("LAS_MANAGER_CONFIG", str(path))
    loaded = load_config()
    assert loaded.resolved_path("cache_dir") == tmp_path / "cache"
    assert loaded.resolved_snapshot_dirs() == (tmp_path / "runs",)


def test_config_params_are_string_array(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('params = ["--devices", 8]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="params must be an array of strings"):
        load_config(path)


def test_config_rclone_params_are_string_array(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('rclone_params = ["--transfers", 512]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="rclone_params must be an array of strings"):
        load_config(path)


@pytest.mark.parametrize("value", ["0", "-1", "true", '"128"'])
def test_config_fiberlet_threads_must_be_positive_integer(tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(f"fiberlet_threads = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fiberlet_threads must be a positive integer"):
        load_config(path)


def test_command_unique_prefix_and_ambiguity():
    assert _expand_command(["sn", "l"]) == ["snapshot", "ls"]
    assert _expand_command(["con", "sh"]) == ["config", "show"]
    with pytest.raises(ValueError, match="ambiguous"):
        _expand_command(["f"])
    assert _expand_command(["fe"]) == ["fetch"]
    assert _expand_command(["fi", "ru"]) == ["fiberlet", "run"]
    assert _expand_command(["completion", "ins"]) == ["completion", "install"]
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_token("f", ("fetch", "foo"))


def test_completion_script_is_generated_without_config(monkeypatch):
    monkeypatch.setenv("LAS_MANAGER_CONFIG", "/does/not/exist")
    assert "complete -F _las_manager_complete las_manager" in _completion_script("bash")
    assert "compdef _las_manager las_manager" in _completion_script("zsh")
    assert main(["completion", "bash"]) == 0


def test_completion_scripts_cover_registry_and_dynamic_selectors():
    for shell in ("bash", "zsh"):
        script = _completion_script(shell)
        assert "_complete-argv" in script
        if shell == "bash":
            assert '"${COMP_WORDS[@]:1}"' in script
        else:
            assert '"${words[@]:2}"' in script


def _fake_las_manager(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True)
    identity = provider_id(path)
    path.write_text(
        "#!/bin/bash\n"
        f"if [[ \"$1\" == _completion-provider-id ]]; then echo {identity}; exit 0; fi\n"
        f"if [[ \"$1\" == _complete-argv ]]; then printf '%s\\tprovider\\n' {value}; exit 0; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_completion_install_dispatches_to_path_selected_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    first = _fake_las_manager(tmp_path / "venv-a" / "bin" / "las_manager", "alpha-volume")
    second = _fake_las_manager(tmp_path / "venv-b" / "bin" / "las_manager", "beta-volume")
    for executable in (first, second):
        identity = provider_id(executable)
        provider = _completion_script(
            "bash", command=str(executable),
            function_name=f"_las_manager_complete_{identity}", register=False,
        )
        loader = install_bash_completion(executable, provider)

    registry = json.loads(
        (tmp_path / "data/las_manager/completions/bash/providers.json").read_text(encoding="utf-8")
    )
    assert set(registry.values()) == {str(first), str(second)}
    subprocess.run(["bash", "-n", str(loader)], check=True)

    script = f'''source {loader!s}
PATH={first.parent}
COMP_WORDS=(las_manager volume prefetch "")
COMP_CWORD=3
_las_manager_completion_dispatch
printf 'first=%s\\n' "${{COMPREPLY[*]}}"
PATH={second.parent}
COMP_WORDS=(las_manager volume prefetch "")
COMP_CWORD=3
_las_manager_completion_dispatch
printf 'second=%s\\n' "${{COMPREPLY[*]}}"
'''
    completed = subprocess.run(["bash", "-c", script], check=True, text=True, capture_output=True)
    assert completed.stdout.splitlines() == ["first=alpha-volume", "second=beta-volume"]

    before = loader.read_bytes()
    identity = provider_id(first)
    install_bash_completion(
        first,
        _completion_script(
            "bash", command=str(first),
            function_name=f"_las_manager_complete_{identity}", register=False,
        ),
    )
    assert loader.read_bytes() == before


def test_completion_executable_identity_canonicalizes_symlinks(tmp_path):
    target = tmp_path / "implementation"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    link = tmp_path / "bin" / "las_manager"
    link.parent.mkdir()
    link.symlink_to(target)
    assert canonical_executable(str(link)) == target.resolve()
    assert provider_id(link) == provider_id(target)


def test_completion_install_cli_defaults_to_bash(tmp_path, monkeypatch, capsys):
    executable = _fake_las_manager(tmp_path / "venv" / "bin" / "las_manager", "value")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(sys, "argv", [str(executable)])
    assert main(["completion", "ins"]) == 0
    loader = tmp_path / "data/bash-completion/completions/las_manager"
    assert loader.is_file()
    assert str(loader) in capsys.readouterr().out


def test_catalog_index_preserves_identity_and_selectors():
    document = sample_catalog()
    raw = json.dumps(document, separators=(",", ":")).encode()
    cache = CatalogCache(document, {"sha256": hashlib.sha256(raw).hexdigest(), "fetched_at": "now"})
    records = index_volumes(cache)
    assert len(records) == 2
    first = records[0]
    assert first.selector == "PHerc0001/20260101000001-2.4um.zarr"
    assert first.s3_url == "s3://public/PHerc0001/volumes/20260101000001-2.4um.zarr/"
    assert first.license["name"] == "CC BY-NC 4.0"
    assert first.catalog_metadata["sha256"] == first.catalog_sha256
    assert first.raw["data"][0]["origins"] == list(first.origins)
    assert resolve_volume(records, first.selector) == first
    assert resolve_volume(records, "20260101000001") == first
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_volume(records, "2026")


def test_catalog_index_tolerates_explicit_null_shape():
    document = sample_catalog()
    first_sample = next(iter(document["samples"].values()))
    first_volume = next(iter(first_sample["volumes"].values()))
    first_volume["properties"]["shape"] = None
    cache = CatalogCache(document, {"sha256": "digest", "fetched_at": "now"})
    records = index_volumes(cache)
    assert records[0].shape == ()


def test_null_shape_volume_renders_unknown():
    document = sample_catalog()
    first_sample = next(iter(document["samples"].values()))
    next(iter(first_sample["volumes"].values()))["properties"]["shape"] = None
    record = index_volumes(CatalogCache(document, {"sha256": "digest"}))[0]
    table = _render_volume_table([record], configured(Path("/tmp")))
    assert "SHAPE" in table
    assert "  -  " in table


def test_volume_table_groups_sorts_and_marks_branches(tmp_path):
    records = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))
    singleton = replace(
        records[0], sample_id="PHerc0002", volume_id="20260101000003",
        long_id="20260101000003-2.4um.zarr",
    )
    table = _render_volume_table([records[1], singleton, records[0]], configured(tmp_path))
    lines = table.splitlines()
    assert lines[0].split() == [
        "SCROLL", "VOLUME", "SHAPE", "VOXEL", "FORMAT", "PREFETCHED", "ORIGINS",
    ]
    assert len(lines[1]) == len(lines[0])
    assert set(lines[1]) <= {"-", " "}
    assert lines[2].startswith("PHerc0001  20260101000001-2.4um.zarr")
    assert lines[3].startswith("└─         20260101000002-2.4um.zarr")
    assert lines[4].startswith("PHerc0002  20260101000003-2.4um.zarr")
    assert table.count("PHerc0001") == 1
    assert all(line.split()[-2:] == ["-", "s3"] for line in lines[2:])
    volume_column = lines[0].index("VOLUME")
    assert all(line.index("2026") == volume_column for line in lines[2:])
    assert lines[3].index("─") == 1
    assert all("    12x   10x    8" in line for line in lines[2:])


def test_volume_table_prefetched_scales_require_chunk_data(tmp_path):
    config = configured(tmp_path)
    record = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    root = volume_cache_root(config, record)
    for level in (2, 10, 3):
        group = root / str(level)
        group.mkdir(parents=True)
        (group / ".zarray").write_text("{}", encoding="utf-8")
    (root / "2" / "0.0.0").write_bytes(b"chunk")
    nested = root / "10" / "0" / "0"
    nested.mkdir(parents=True)
    (nested / "0").write_bytes(b"chunk")
    table = _render_volume_table([record], config)
    row = table.splitlines()[2]
    assert "  2,10        " in row
    assert "3" not in row.split("2,10", 1)[1].split("s3", 1)[0]


def test_volume_table_empty_and_ascii_fallback(tmp_path):
    config = configured(tmp_path)
    empty = _render_volume_table([], config, unicode_tree=False)
    assert len(empty.splitlines()) == 2
    records = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))
    ascii_table = _render_volume_table(records, config, unicode_tree=False)
    assert "\\-" in ascii_table
    assert "20260101000001-2.4um.zarr" in ascii_table
    assert "└" not in ascii_table


def test_contextual_help_uses_longest_understood_prefix():
    assert _rewrite_contextual_help(["help"]) == ["--help"]
    assert _rewrite_contextual_help(["vol", "help"]) == ["volume", "--help"]
    assert _rewrite_contextual_help(["vol", "pre", "help"]) == ["volume", "prefetch", "--help"]
    assert _rewrite_contextual_help(["volume", "nonsense", "help"]) == ["volume", "--help"]
    assert _rewrite_contextual_help(["volume", "prefetch", "anything", "help"]) == ["volume", "prefetch", "--help"]
    forwarded = ["inference", "run", "snap", "vol", "2", "--", "help"]
    assert _rewrite_contextual_help(forwarded) == forwarded
    with pytest.raises(ValueError, match="unknown command"):
        _rewrite_contextual_help(["nonsense", "help"])


def test_contextual_help_prints_selected_parser_without_config(monkeypatch, capsys):
    monkeypatch.setenv("LAS_MANAGER_CONFIG", "/does/not/exist")
    with pytest.raises(SystemExit) as exit_info:
        main(["vol", "pre", "ignored-positional", "help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: las_manager volume prefetch" in output
    assert "--workers" in output


def test_contextual_completion_commands_options_and_values(tmp_path, monkeypatch):
    from lasagna.manager import completion

    config = configured(tmp_path)
    catalog_cache = CatalogCache(sample_catalog(), {"sha256": "digest"})
    records = index_volumes(catalog_cache)
    monkeypatch.setattr(completion, "_volume_records", lambda _config: records)
    monkeypatch.setattr(completion, "_cached_catalog", lambda _config: catalog_cache)
    values = lambda words: [value for value, _description in contextual_candidates(config, words)]
    assert "volume" in values(["vol"])
    assert values(["vol", "pr"]) == ["prefetch"]
    assert records[0].selector in values(["vol", "pre", ""])
    assert values(["volume", "ls", "--backend=f"]) == []
    assert values(["snapshot", "ls", "--backend", "f"]) == ["fiber3d"]
    assert values(["snapshot", "ls", "--backend=f"]) == ["--backend=fiber3d"]
    assert "--sample" in values(["volume", "ls", "--"])
    assert "PHerc0001" in values(["volume", "ls", "--sample", ""])
    assert "--format=uint8" in values(["volume", "ls", "--format="])
    assert values(["inference", "run", "snap", "vol", "2", "--", "anything"]) == []


def test_hidden_contextual_completion_transports_option_tokens(monkeypatch, capsys):
    monkeypatch.setenv("LAS_MANAGER_CONFIG", "/does/not/exist")
    assert main(["_complete-argv", "snapshot", "ls", "--backend", "f"]) == 0
    assert capsys.readouterr().out.splitlines() == ["fiber3d\toption value"]


def test_contextual_completion_discovers_only_local_scales(tmp_path, monkeypatch):
    from lasagna.manager import completion
    from lasagna.manager.prefetch import volume_cache_root

    config = configured(tmp_path)
    record = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    monkeypatch.setattr(completion, "_volume_records", lambda _config: [record])
    words = ["volume", "prefetch", record.selector, ""]
    assert contextual_candidates(config, words) == []
    root = volume_cache_root(config, record)
    root.mkdir(parents=True)
    (root / ".zattrs").write_text(
        json.dumps({"multiscales": [{"datasets": [{"path": "0"}, {"path": "2"}]}]}),
        encoding="utf-8",
    )
    level = root / "1"
    level.mkdir()
    (level / ".zarray").write_text("{}", encoding="utf-8")
    assert [value for value, _ in contextual_candidates(config, words)] == ["0", "1", "2"]


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.headers = FakeHeaders({"ETag": '"v1"', "Last-Modified": "today"})

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_catalog_refresh_cache_and_offline_fallback(tmp_path, monkeypatch):
    config = configured(tmp_path)
    body = json.dumps(sample_catalog()).encode()
    monkeypatch.setattr(catalog, "urlopen", lambda request, timeout: FakeResponse(body))
    fetched = get_catalog(config, force_refresh=True, now=100.0)
    assert fetched.metadata["etag"] == '"v1"'
    assert len(index_volumes(fetched)) == 2
    monkeypatch.setattr(catalog, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    fallback = get_catalog(config, force_refresh=True, now=200.0)
    assert fallback.warning and "offline" in fallback.warning
    assert fallback.metadata["sha256"] == fetched.metadata["sha256"]


def test_catalog_never_networks_when_completion_style_cached_only(tmp_path, monkeypatch):
    config = configured(tmp_path)
    body = json.dumps(sample_catalog()).encode()
    monkeypatch.setattr(catalog, "urlopen", lambda request, timeout: FakeResponse(body))
    get_catalog(config, force_refresh=True, now=100.0)
    monkeypatch.setattr(catalog, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))
    assert get_catalog(config, allow_network=False, now=999999.0).document["samples"]


def test_snapshot_roots_metadata_cache_and_selector(tmp_path):
    torch = pytest.importorskip("torch")
    runs = tmp_path / "runs"
    snapshot_dir = runs / "run-a" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    checkpoint = snapshot_dir / "best.pt"
    torch.save({
        "model": {"weight": torch.ones(1)},
        "step": 42,
        "metric": 0.25,
        "metric_name": "test/loss_total",
        "config": {
            "patch_shape_zyx": [128, 128, 128],
            "model_3d": {"direction_branch_count": 2, "output_channels": 14},
            "training": {"mixed_precision": "bf16"},
            "atlas_model_id": "20260806120000",
        },
    }, checkpoint)
    assert discover_snapshot_paths((runs,)) == [("run-a", checkpoint.resolve())]
    config = configured(tmp_path, snapshot_dirs=(runs, snapshot_dir, runs / "run-a"))
    records = index_snapshots(config)
    assert len(records) == 1
    record = records[0]
    assert record.selector == "fiber3d/run-a/best.pt"
    assert record.step == 42
    assert record.patch_shape == (128, 128, 128)
    assert record.option_count == 2
    assert record.precision_policy == "bf16"
    assert record.task == "lasagna"
    assert record.atlas_model_id == "20260806120000"
    assert resolve_snapshot(records, "run-a/b") == record
    checkpoint.unlink()
    assert index_snapshots(config, cached_only=True) == []


def test_snapshot_completion_rescans_uncached_files_without_loading_them(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    snapshots = runs / "new-run" / "snapshots"
    snapshots.mkdir(parents=True)
    checkpoint = snapshots / "best.pt"
    checkpoint.write_bytes(b"not a checkpoint; completion must not load it")
    config = configured(tmp_path, snapshot_dirs=(runs,))

    assert completion_snapshot_candidates(config) == [
        ("new-run/best.pt", "unindexed snapshot"),
    ]
    assert not (tmp_path / "cache" / "snapshots" / "index.json").exists()
    values = [
        value for value, _description
        in contextual_candidates(config, ["inference", "run", "new"])
    ]
    assert values == ["new-run/best.pt"]


def test_lasagna_snapshot_discovery_is_namespaced_and_extracts_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    snapshots = tmp_path / "runs" / "las-run" / "snapshots"
    snapshots.mkdir(parents=True)
    checkpoint = snapshots / "model_best.pt"
    torch.save({
        "state_dict": {"shared_encoder.stages.0.weight": torch.ones(1)},
        "patch_size": 256,
        "norm_type": "group",
        "upsample_mode": "trilinear",
        "precision": "fp16",
        "val_loss": 0.125,
        "atlas_model_id": "20260806123000",
    }, checkpoint)
    config = configured(tmp_path, snapshot_dirs=(tmp_path / "runs",))
    record = index_snapshots(config)[0]
    assert record.backend == "lasagna"
    assert record.selector == "lasagna/las-run/model_best.pt"
    assert record.patch_shape == (256, 256, 256)
    assert record.architecture == "lasagna_3d"
    assert record.metric_name == "validation/loss"
    assert record.metric_value == 0.125
    assert record.precision_policy == "fp16"
    assert record.atlas_model_id == "20260806123000"


def test_cli_config_init_and_volume_list(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("LAS_MANAGER_CONFIG", str(config_path))
    assert main(["con", "init"]) == 0
    text = config_path.read_text(encoding="utf-8").replace('cache_dir = ""', f'cache_dir = "{tmp_path / "cache"}"')
    config_path.write_text(text, encoding="utf-8")
    config = load_config()
    body = json.dumps(sample_catalog()).encode()
    monkeypatch.setattr(catalog, "urlopen", lambda request, timeout: FakeResponse(body))
    get_catalog(config, force_refresh=True)
    assert main(["vol", "l", "--sample", "PHerc0001"]) == 0
    output = capsys.readouterr().out
    assert "SCROLL" in output
    assert output.count("PHerc0001") == 1
    assert "└─" in output
    assert "20260101000002-2.4um.zarr" in output
    assert main(["volume", "ls", "--sample", "PHerc0001", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["long_id"] for item in payload] == [
        "20260101000001-2.4um.zarr", "20260101000002-2.4um.zarr",
    ]


def test_prefetch_reuses_downloader_and_root_convention(tmp_path, monkeypatch):
    config = configured(tmp_path)
    record = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    calls = []
    monkeypatch.setattr(
        "lasagna.scripts.download_omezarr.download",
        lambda **kwargs: calls.append(kwargs) or 0,
    )
    result = prefetch_volume(config, record, 2, workers=17, remote_inventory=False)
    assert result == volume_cache_root(config, record) / "2"
    assert calls == [{
        "source": record.s3_url,
        "dest": str(volume_cache_root(config, record)),
        "scales": [2],
        "workers": 17,
        "anon": True,
        "remote_inventory": False,
    }]


class FakeTmux:
    def __init__(self, sessions=()):
        self.sessions = set(sessions)
        self.created = []

    def has_session(self, session):
        return session in self.sessions

    def create(self, session, window, argv, *, run_uuid):
        self.sessions.add(session)
        self.created.append((session, window, list(argv), run_uuid))
        return "@99"

    def has_window(self, window_id):
        return window_id == "@99"

    def window_matches(self, window_id, run_uuid):
        return window_id == "@99"


def _completed_prediction_run(
    config: ManagerConfig, *, name: str, kind: str, channels: list[str],
    sample_id: str = "PHerc0001", volume_id: str = "20260101000001",
) -> Path:
    run_dir = Path(config.output_dir) / name
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    manifest = artifacts / f"{name}.lasagna.json"
    manifest.write_text(json.dumps({
        "version": 2,
        "source_to_base": 1.0,
        "base_shape_zyx": [100, 80, 60],
        "groups": {
            channel: {
                "zarr": f"{channel}.ome.zarr/2",
                "scaledown": 2,
                "channels": [channel],
            }
            for channel in channels
        },
    }), encoding="utf-8")
    atomic_json(artifacts / "inference.json", {
        "schema_version": 1,
        "artifact_kind": kind,
        "status": "completed",
        "run_uuid": f"{name}-uuid",
        "source": {
            "sample_id": sample_id,
            "volume_id": volume_id,
            "license": {"name": "CC BY-NC 4.0"},
            "requested_group": 1,
        },
        "source_scale": {
            "requested_group": 1,
            "base_shape_zyx": [100, 80, 60],
            "source_to_base_factor": 2,
        },
        "inference": {"crop_xyzwhd_base": None},
        "model": {
            "atlas_model_id": f"{name}-model", "run": f"{name}-training",
            "snapshot": f"{name}-training/snapshots/best.pt",
            "sha256": "a" * 64,
        },
        "artifacts": [{"kind": "manifest", "path": manifest.name}],
    })
    atomic_json(run_dir / "metadata.json", {
        "run_uuid": f"{name}-uuid", "run_name": name,
        "job_kind": "inference", "active_lifecycle_phase": "inference",
        "artifact_kind": kind, "status": "completed",
        "created_at": "2026-08-23T00:00:00Z",
        "artifacts": {
            "root": "artifacts", "provenance": "artifacts/inference.json",
        },
        "lifecycle": {"inference": "completed"},
    })
    return run_dir


def test_fiberlet_launch_uses_shared_runner_and_portable_source_identity(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    binary = tmp_path / "bin" / "vc_fiberlets"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    config = replace(
        configured(tmp_path), fiberlet_binary=str(binary),
        fiberlet_threads=32,
    )
    _completed_prediction_run(
        config, name="fiber-source", kind="fiber3d-prediction",
        channels=["presence", "nx", "ny"],
    )
    _completed_prediction_run(
        config, name="normal-source", kind="lasagna", channels=["nx", "ny"],
    )
    fake = FakeTmux()
    run_dir = launch_fiberlet(
        config, "fiber-source", "normal-source",
        original_argv=["fiberlet", "run", "fiber-source", "normal-source"],
        extra_args=("--threads", "8"), tmux=fake,
    )
    record = json.loads((run_dir / "metadata.json").read_text())
    command = json.loads((run_dir / "command.json").read_text())
    context = json.loads((run_dir / "source_context.json").read_text())
    assert record["job_kind"] == "fiberlet"
    assert record["active_lifecycle_phase"] == "fiberlet_preprocess"
    assert fake.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]
    assert command["resolved_argv"][:2] == [str(binary), "preprocess-volume"]
    assert command["resolved_argv"][-4:] == ["--threads", "32", "--threads", "8"]
    assert str(run_dir / "cache") in command["resolved_argv"][command["resolved_argv"].index("--anchor-cache") + 1]
    assert str(run_dir / "cache") not in json.dumps(context)
    assert "manifest_path" not in json.dumps(context)
    assert context["source_volume"] == {
        "sample_id": "PHerc0001", "volume_id": "20260101000001",
    }

    dataset = run_dir / "artifacts" / "fiberlets.zarr"
    dataset.mkdir()
    atomic_json(dataset / ".zattrs", {
        "vc_format": "fiberlet_dataset", "dataset_kind": "combined",
        "format_version": 2, "dataset_fingerprint": "b" * 64,
        "algorithm_fingerprint": "fnv1a64:0123456789abcdef",
        "processing": {"contract_version": 2},
        "sources": {
            name: {
                "run_uuid": dependency["run_uuid"],
                "manifest_sha256": dependency["manifest_sha256"],
            }
            for name, dependency in record["dependencies"].items()
        },
    })
    (dataset / ".zgroup").write_text("{}", encoding="utf-8")
    for relative in ("anchors/.zarray", "prefix/.zarray", "routes/.zarray"):
        path = dataset / relative
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
    finalize_fiberlet_provenance(run_dir, record)
    portable = json.loads((run_dir / "artifacts" / "fiberlets.json").read_text())
    assert portable["artifact_kind"] == "fiberlets"
    assert portable["artifacts"] == [{
        "kind": "fiberlet-zarr", "path": "fiberlets.zarr",
        "metadata_sha256": hashlib.sha256((dataset / ".zattrs").read_bytes()).hexdigest(),
    }]
    assert str(tmp_path) not in json.dumps(portable)

    record.update(status="interrupted", ended_at="2026-08-23T01:00:00Z")
    record["lifecycle"]["fiberlet_preprocess"] = "interrupted"
    atomic_json(run_dir / "metadata.json", record)
    resumed_tmux = FakeTmux()
    assert resume_fiberlet(config, record["run_name"], tmux=resumed_tmux) == run_dir
    resumed_command = json.loads((run_dir / "command.json").read_text())
    assert resumed_command == command
    assert resumed_tmux.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]


def test_fiberlet_inputs_reject_different_source_volumes(tmp_path):
    config = configured(tmp_path)
    _completed_prediction_run(
        config, name="fiber-source", kind="fiber3d-prediction",
        channels=["presence", "nx", "ny"],
    )
    _completed_prediction_run(
        config, name="normal-source", kind="lasagna", channels=["nx", "ny"],
        volume_id="other",
    )
    with pytest.raises(ValueError, match="same source volume_id"):
        resolve_fiberlet_inputs(config, "fiber-source", "normal-source")


def test_fiberlet_completion_filters_inputs_by_role(tmp_path):
    config = configured(tmp_path)
    _completed_prediction_run(
        config, name="fiber-source", kind="fiber3d-prediction",
        channels=["presence", "nx", "ny"],
    )
    _completed_prediction_run(
        config, name="normal-source", kind="lasagna", channels=["nx", "ny"],
    )
    first = [value for value, _ in contextual_candidates(config, ["fiberlet", "run", ""])]
    second = [
        value for value, _ in contextual_candidates(
            config, ["fiberlet", "run", "fiber-source", ""],
        )
    ]
    assert first == ["fiber-source"]
    assert second == ["normal-source"]


def test_fiberlet_launch_rejects_manager_owned_native_options(tmp_path):
    binary = tmp_path / "vc_fiberlets"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    config = replace(configured(tmp_path), fiberlet_binary=str(binary))
    _completed_prediction_run(
        config, name="fiber-source", kind="fiber3d-prediction",
        channels=["presence", "nx", "ny"],
    )
    _completed_prediction_run(
        config, name="normal-source", kind="lasagna", channels=["nx", "ny"],
    )
    with pytest.raises(ValueError, match="manager-owned.*--anchor-cache"):
        launch_fiberlet(
            config, "fiber-source", "normal-source",
            original_argv=["fiberlet", "run"],
            extra_args=("--anchor-cache=/tmp/not-manager-owned",),
            tmux=FakeTmux(),
        )


def _snapshot_and_config(tmp_path: Path):
    torch = pytest.importorskip("torch")
    snapshots = tmp_path / "runs" / "run one" / "snapshots"
    snapshots.mkdir(parents=True)
    checkpoint = snapshots / "best model.pt"
    torch.save({"model": {"w": torch.ones(1)}, "step": 8, "config": {"patch_shape_zyx": [64, 64, 64]}}, checkpoint)
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    config = configured(tmp_path, snapshot_dirs=(tmp_path / "runs",))
    return config, index_snapshots(config)[0]


def test_launch_writes_backend_neutral_record_and_argv(tmp_path):
    config, snapshot = _snapshot_and_config(tmp_path)
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest", "fetched_at": "now"}))[0]
    fake = FakeTmux()
    run_dir = launch_inference(
        config, snapshot, volume, 2,
        original_argv=["inference", "run", snapshot.selector, volume.selector, "2"],
        extra_args=["--devices", "all"], tmux=fake,
    )
    metadata = json.loads((run_dir / "metadata.json").read_text())
    command = json.loads((run_dir / "command.json").read_text())
    assert metadata["status"] == "created"
    assert metadata["lifecycle"] == {
        "prefetch": "pending", "inference": "created", "staging_upload": "not_started",
        "atlas_ingest": "not_started", "atlas_publication": "not_started",
    }
    assert command["prefetch"]["scale"] == 2
    assert command["prefetch"]["workers"] == 64
    assert metadata["source"]["scale"] == 2
    assert metadata["artifacts"]["manifest"].startswith("artifacts/")
    assert command["resolved_argv"][-2:] == ["--devices", "all"]
    assert any(value.endswith("best model.pt") for value in command["resolved_argv"])
    assert not any(value.endswith("fiber_config.json") for value in command["resolved_argv"])
    assert "--provenance-context" in command["resolved_argv"]
    assert "--no-download" in command["resolved_argv"]
    context = json.loads((run_dir / "provenance_context.json").read_text())
    assert context["run_uuid"] == metadata["run_uuid"]
    assert context["source"]["requested_group"] == 2
    assert "path" not in context["model"]
    assert context["model"]["snapshot"] == "run one/snapshots/best model.pt"
    assert context["model"]["architecture"] == "fiber3d/unet"
    assert fake.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]
    assert run_dir.name.startswith(f"{volume.sample_id}-{volume.volume_id}-las-sd2-")
    assert snapshot.run not in run_dir.name
    assert metadata["source"]["volume"]["selector"] == volume.selector
    assert metadata["tmux_window_name"].startswith(f"inf-{volume.sample_id}-")
    assert fake.created[0][1] == metadata["tmux_window_name"]


def test_run_config_params_precede_explicit_backend_overrides(tmp_path):
    config, snapshot = _snapshot_and_config(tmp_path)
    config = replace(config, params=("--tile-size", "512", "--devices", "all"))
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    run_dir = launch_inference(
        config, snapshot, volume, 1, original_argv=["inference", "run"],
        extra_args=("--tile-size", "256", "--devices", "cuda:1"), tmux=FakeTmux(),
    )
    argv = json.loads((run_dir / "command.json").read_text())["resolved_argv"]
    assert argv[-6:] == [
        "--tile-size", "512",
        "--tile-size", "256", "--devices", "cuda:1",
    ]

    second = launch_inference(
        config, snapshot, volume, 1, original_argv=["inference", "run"],
        extra_args=("--device", "cuda:1"), tmux=FakeTmux(),
    )
    argv = json.loads((second / "command.json").read_text())["resolved_argv"]
    assert "--devices" not in argv
    assert argv[-2:] == ["--device", "cuda:1"]


@pytest.mark.parametrize("backend", ["fiber3d", "lasagna"])
def test_no_prefetch_delegates_downloads_to_backend(tmp_path, backend):
    config, fiber_snapshot = _snapshot_and_config(tmp_path)
    snapshot = fiber_snapshot
    if backend == "lasagna":
        snapshot = fiber_snapshot.__class__(**{
            **fiber_snapshot.__dict__,
            "backend": "lasagna",
            "selector": "lasagna/run one/best model.pt",
            "architecture": "lasagna_3d",
        })
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    run_dir = launch_inference(
        config, snapshot, volume, 1, original_argv=["inference", "run"],
        prefetch=False, download_workers=321, tmux=FakeTmux(),
    )
    command = json.loads((run_dir / "command.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert "--no-download" not in command["resolved_argv"]
    worker_index = command["resolved_argv"].index("--download-workers")
    assert command["resolved_argv"][worker_index + 1] == "321"
    assert command["prefetch"] is None
    assert metadata["lifecycle"]["prefetch"] == "skipped"
    cache_attrs = json.loads((volume_cache_root(config, volume) / ".zattrs").read_text())
    assert cache_attrs["_download"] == {"source": volume.s3_url, "anon": True}


def test_no_prefetch_respects_explicit_backend_no_download(tmp_path):
    config, snapshot = _snapshot_and_config(tmp_path)
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    run_dir = launch_inference(
        config, snapshot, volume, 1, original_argv=["inference", "run"],
        prefetch=False, extra_args=("--no-download",), tmux=FakeTmux(),
    )
    argv = json.loads((run_dir / "command.json").read_text())["resolved_argv"]
    assert argv[-1] == "--no-download"


@pytest.mark.parametrize("backend", ["fiber3d", "lasagna"])
def test_live_fetch_skips_bulk_prefetch_and_records_shared_backend_options(tmp_path, backend):
    config, fiber_snapshot = _snapshot_and_config(tmp_path)
    snapshot = fiber_snapshot
    if backend == "lasagna":
        snapshot = fiber_snapshot.__class__(**{
            **fiber_snapshot.__dict__, "backend": "lasagna",
            "selector": "lasagna/run one/best model.pt", "architecture": "lasagna_3d",
        })
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    run_dir = launch_inference(
        config, snapshot, volume, 1, original_argv=["inference", "run"],
        prefetch=False, live_fetch=True, live_cache_gib=12.5,
        live_fetch_ahead_tiles=4321, download_workers=123, tmux=FakeTmux(),
    )
    command = json.loads((run_dir / "command.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    argv = command["resolved_argv"]
    assert command["prefetch"] is None
    assert "--no-download" not in argv
    assert argv[argv.index("--live-cache-gib") + 1] == "12.5"
    assert argv[argv.index("--live-fetch-ahead-tiles") + 1] == "4321"
    assert argv[argv.index("--download-workers") + 1] == "123"
    assert metadata["live_fetch"] == {
        "enabled": True, "cache_target_gib": 12.5,
        "lookahead_tiles": 4321, "download_workers": 123,
    }
    cache_attrs = json.loads((volume_cache_root(config, volume) / ".zattrs").read_text())
    assert cache_attrs["_download"]["source"] == volume.s3_url


def test_live_fetch_rejects_backend_no_download_before_source_initialization(tmp_path):
    config, snapshot = _snapshot_and_config(tmp_path)
    config = replace(config, params=("--no-download",))
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest"}))[0]
    with pytest.raises(ValueError, match="conflicts with --no-download"):
        launch_inference(
            config, snapshot, volume, 1, original_argv=["inference", "run"],
            prefetch=False, live_fetch=True, tmux=FakeTmux(),
        )
    assert not (volume_cache_root(config, volume) / ".zattrs").exists()


def test_lasagna_launch_reuses_shared_run_and_tmux_workflow(tmp_path):
    config, fiber_snapshot = _snapshot_and_config(tmp_path)
    snapshot = fiber_snapshot.__class__(**{
        **fiber_snapshot.__dict__,
        "backend": "lasagna",
        "selector": "lasagna/run one/best model.pt",
        "architecture": "lasagna_3d",
    })
    volume = index_volumes(CatalogCache(sample_catalog(), {"sha256": "digest", "fetched_at": "now"}))[0]
    fake = FakeTmux()
    run_dir = launch_inference(
        config, snapshot, volume, 1,
        original_argv=["inference", "run", snapshot.selector, volume.selector, "1"],
        extra_args=["--devices", "all"], tmux=fake,
    )
    metadata = json.loads((run_dir / "metadata.json").read_text())
    command = json.loads((run_dir / "command.json").read_text())["resolved_argv"]
    assert metadata["backend"] == "lasagna"
    assert metadata["artifact_kind"] == "lasagna"
    assert metadata["lifecycle"]["atlas_ingest"] == "not_started"
    assert command[1:4] == ["-m", "preprocess_cos_omezarr", "predict3d"]
    assert command[command.index("--input") + 1].endswith("/1")
    assert "--provenance-context" in command
    assert "--no-download" in command
    assert command[-2:] == ["--devices", "all"]
    assert fake.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]


def test_inference_run_returns_after_detached_launch_without_foreground_prefetch(
    tmp_path, monkeypatch, capsys,
):
    config, snapshot = _snapshot_and_config(tmp_path)
    cache = CatalogCache(sample_catalog(), {"sha256": "digest", "fetched_at": "now"})
    volume = index_volumes(cache)[0]
    calls = []
    monkeypatch.setattr(manager_cli, "load_config", lambda: config)
    monkeypatch.setattr(manager_cli, "get_catalog", lambda _config: cache)
    monkeypatch.setattr(manager_cli, "index_snapshots", lambda _config: [snapshot])
    monkeypatch.setattr(
        manager_cli, "prefetch_volume",
        lambda *_args, **_kwargs: pytest.fail("prefetch ran in the manager process"),
    )

    def fake_launch(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "reserved-run"

    monkeypatch.setattr(manager_cli, "launch_inference", fake_launch)
    assert main([
        "inference", "run", snapshot.selector, volume.selector, "2",
        "--download-workers", "511",
    ]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "reserved-run")
    assert calls[0][1]["prefetch"] is True
    assert calls[0][1]["download_workers"] == 511
    assert main([
        "inference", "run", snapshot.selector, volume.selector, "2",
        "--no-prefetch", "--download-workers", "333",
    ]) == 0
    capsys.readouterr()
    assert calls[1][1]["prefetch"] is False
    assert calls[1][1]["download_workers"] == 333


def test_reconcile_marks_dead_running_record_interrupted(tmp_path):
    config = configured(tmp_path)
    run_dir = Path(config.output_dir) / "dead"
    record = {
        "run_name": "dead", "run_uuid": "uuid", "status": "running",
        "pid": 99999999, "process_start_time": "1", "tmux_session": "las-dead",
        "lifecycle": {"inference": "running"}, "created_at": "2026-01-01T00:00:00Z",
    }
    atomic_json(run_dir / "metadata.json", record)
    reconciled = reconcile_runs(config, FakeTmux())
    assert reconciled[0][1]["status"] == "interrupted"
    assert reconciled[0][1]["lifecycle"]["inference"] == "interrupted"


def test_runner_captures_log_and_failed_exit(tmp_path):
    run_dir = tmp_path / "run with spaces"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"inference": "created"},
        "started_at": None, "ended_at": None,
    })
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [sys.executable, "-c", "print('hello runner'); raise SystemExit(7)"]
    })
    assert runner_main([str(run_dir)]) == 7
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 7
    assert "hello runner" in (run_dir / "run.log").read_text()


def test_runner_tees_inference_output_to_log_and_terminal(tmp_path, capfd):
    run_dir = tmp_path / "tee-output"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"inference": "created"},
        "started_at": None, "ended_at": None,
    })
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [
            sys.executable, "-c",
            "import sys; sys.stdout.write('progress\\r42%'); sys.stdout.flush(); raise SystemExit(3)",
        ],
    })
    assert runner_main([str(run_dir)]) == 3
    assert b"progress\r42%" in (run_dir / "run.log").read_bytes()
    assert "progress\r42%" in capfd.readouterr().out


def test_runner_prefetches_before_inference_and_preserves_options(tmp_path, monkeypatch, capfd):
    run_dir = tmp_path / "detached"
    marker = run_dir / "prefetched"
    provenance = run_dir / "artifacts" / "inference.json"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"prefetch": "pending", "inference": "created"},
        "started_at": None, "ended_at": None,
        "artifacts": {"provenance": "artifacts/inference.json", "inventory": []},
    })
    request = {
        "version": 1, "source": "s3://bucket/volume.zarr",
        "destination": str(run_dir / "cache"), "scale": 3, "workers": 511,
        "anon": True, "remote_inventory": False,
    }

    def fake_prefetch(value):
        assert value == request
        print("prefetch stdout progress", flush=True)
        print("prefetch stderr progress", file=sys.stderr, flush=True)
        marker.write_text("ready", encoding="utf-8")
        return Path(value["destination"]) / str(value["scale"])

    script = (
        "from pathlib import Path; import json; "
        f"assert Path({str(marker)!r}).is_file(); "
        f"p=Path({str(provenance)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'status':'completed','artifacts':[]}))"
    )
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [sys.executable, "-c", script], "prefetch": request,
    })
    monkeypatch.setattr("lasagna.manager.runner.execute_prefetch_request", fake_prefetch)

    assert runner_main([str(run_dir)]) == 0
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["lifecycle"]["prefetch"] == "completed"
    assert metadata["prefetch"]["error"] is None
    log = (run_dir / "run.log").read_text()
    assert "prefetch stdout progress" in log
    assert "prefetch stderr progress" in log
    assert "prefetch completed" in log
    captured = capfd.readouterr()
    assert "prefetch stdout progress" in captured.out
    assert "prefetch stderr progress" in captured.err


def test_runner_prefetch_failure_never_starts_inference(tmp_path, monkeypatch):
    run_dir = tmp_path / "prefetch-failure"
    marker = run_dir / "inference-started"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"prefetch": "pending", "inference": "created"},
        "started_at": None, "ended_at": None,
    })
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        "prefetch": {"version": 1},
    })
    monkeypatch.setattr(
        "lasagna.manager.runner.execute_prefetch_request",
        lambda _request: (_ for _ in ()).throw(RuntimeError("download exploded")),
    )

    assert runner_main([str(run_dir)]) == 1
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["lifecycle"]["prefetch"] == "failed"
    assert metadata["lifecycle"]["inference"] == "created"
    assert "download exploded" in metadata["prefetch"]["error"]
    assert not marker.exists()


def test_runner_copies_direct_inference_inventory(tmp_path):
    run_dir = tmp_path / "complete"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"inference": "created"},
        "started_at": None, "ended_at": None,
        "artifacts": {"provenance": "artifacts/inference.json", "inventory": []},
    })
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [sys.executable, "-c", "raise SystemExit(0)"]
    })
    atomic_json(run_dir / "artifacts" / "inference.json", {
        "status": "completed", "artifacts": [{"kind": "manifest", "path": "x.json"}],
        "inference": {"live_fetch": {
            "enabled": True, "cache_target_gib": 10240.0,
            "final_stats": {"downloaded_chunks": 12},
        }},
    })

    assert runner_main([str(run_dir)]) == 0
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["artifacts"]["inventory"] == [{"kind": "manifest", "path": "x.json"}]
    assert metadata["live_fetch"]["final_stats"]["downloaded_chunks"] == 12


def test_runner_rejects_zero_exit_without_completed_provenance(tmp_path):
    run_dir = tmp_path / "missing-provenance"
    atomic_json(run_dir / "metadata.json", {
        "status": "created", "lifecycle": {"inference": "created"},
        "started_at": None, "ended_at": None,
        "artifacts": {"provenance": "artifacts/inference.json", "inventory": []},
    })
    atomic_json(run_dir / "command.json", {
        "resolved_argv": [sys.executable, "-c", "raise SystemExit(0)"]
    })

    assert runner_main([str(run_dir)]) == 1
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 1
    assert metadata["lifecycle"]["inference"] == "failed"
    assert "was not created" in metadata["completion_error"]


def test_tmux_inside_links_adjacent_without_renaming_source(monkeypatch):
    calls = []
    linked = False
    tmux = Tmux("fake-tmux")

    class Result:
        def __init__(self, stdout=""):
            self.stdout = stdout
            self.returncode = 0

    def fake_run(args, **kwargs):
        nonlocal linked
        calls.append(args)
        if "link-window" in args:
            linked = True
        if "show-options" in args:
            return Result("run-uuid\n")
        if "list-windows" in args:
            suffix = "main\t5\t@9\n" if linked else ""
            return Result("las-example\t0\t@9\n" + suffix)
        if "display-message" in args and "#{session_name}" in args:
            return Result("main\n")
        if "display-message" in args and "#{window_index}" in args:
            return Result("4\n")
        if "display-message" in args:
            return Result("@9\n")
        return Result()

    monkeypatch.setattr("lasagna.manager.tmux.subprocess.run", fake_run)
    assert tmux.attach(
        "las-example", window_id="@9", run_uuid="run-uuid",
        environ={"TMUX": "yes"},
    ) == "@9"
    assert ["fake-tmux", "link-window", "-a", "-s", "@9", "-t", "main:4"] in calls
    assert ["fake-tmux", "select-window", "-t", "main:5"] in calls
    assert not any("rename-window" in call for call in calls)


def test_tmux_create_atomically_captures_and_tags_window(monkeypatch):
    calls = []
    tmux = Tmux("fake-tmux")

    class Result:
        def __init__(self, stdout="", returncode=0):
            self.stdout = stdout
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(args)
        if "has-session" in args:
            return Result(returncode=1)
        if "new-session" in args:
            return Result("@42\n")
        return Result()

    monkeypatch.setattr("lasagna.manager.tmux.subprocess.run", fake_run)
    assert tmux.create("las-run", "inference", ["python", "run.py"], run_uuid="uuid") == "@42"
    assert [
        "fake-tmux", "new-session", "-d", "-P", "-F", "#{window_id}",
        "-s", "las-run", "-n", "inference", "python", "run.py",
    ] in calls
    assert [
        "fake-tmux", "set-option", "-w", "-t", "@42",
        "@las_manager_run_uuid", "uuid",
    ] in calls
