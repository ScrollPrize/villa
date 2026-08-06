from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lasagna.manager import catalog
from lasagna.manager.catalog import CatalogCache, get_catalog, index_volumes, resolve_volume
from lasagna.manager.cli import COMMANDS, _completion_script, _expand_command, _resolve_token, main
from lasagna.manager.completion import canonical_executable, install_bash_completion, provider_id
from lasagna.manager.config import ManagerConfig, initialize_config, load_config
from lasagna.manager.prefetch import prefetch_volume, volume_cache_root
from lasagna.manager.runner import main as runner_main
from lasagna.manager.runs import atomic_json, launch_inference, read_runs, reconcile_runs
from lasagna.manager.snapshots import discover_snapshot_paths, index_snapshots, resolve_snapshot
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


def test_command_unique_prefix_and_ambiguity():
    assert _expand_command(["sn", "l"]) == ["snapshot", "ls"]
    assert _expand_command(["con", "sh"]) == ["config", "show"]
    assert _expand_command(["f"]) == ["fetch"]
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
        for root in {command[0] for command in COMMANDS}:
            assert root in script
        if shell == "bash":
            assert "_complete volume" in script
            assert "_complete snapshot" in script
            assert "_complete run" in script
            assert 'completion::*) words="bash install zsh"' in script
        else:
            assert "_las_manager_dynamic volume" in script
            assert "_las_manager_dynamic snapshot" in script
            assert "_las_manager_dynamic run" in script


def _fake_las_manager(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True)
    identity = provider_id(path)
    path.write_text(
        "#!/bin/bash\n"
        f"if [[ \"$1\" == _completion-provider-id ]]; then echo {identity}; exit 0; fi\n"
        f"if [[ \"$1\" == _complete ]]; then printf '%s\\tprovider\\n' {value}; exit 0; fi\n"
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
    assert "PHerc0001/20260101000001-2.4um.zarr" in capsys.readouterr().out


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

    def create(self, session, window, argv):
        self.sessions.add(session)
        self.created.append((session, window, list(argv)))


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
        "inference": "created", "staging_upload": "not_started",
        "atlas_ingest": "not_started", "atlas_publication": "not_started",
    }
    assert metadata["source"]["scale"] == 2
    assert metadata["artifacts"]["manifest"].startswith("artifacts/")
    assert command["resolved_argv"][-2:] == ["--devices", "all"]
    assert any(value.endswith("best model.pt") for value in command["resolved_argv"])
    assert not any(value.endswith("fiber_config.json") for value in command["resolved_argv"])
    assert "--provenance-context" in command["resolved_argv"]
    context = json.loads((run_dir / "provenance_context.json").read_text())
    assert context["run_uuid"] == metadata["run_uuid"]
    assert context["source"]["requested_group"] == 2
    assert "path" not in context["model"]
    assert fake.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]


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
    assert command[-2:] == ["--devices", "all"]
    assert fake.created[0][2][1:3] == ["-m", "lasagna.manager.runner"]


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
    })

    assert runner_main([str(run_dir)]) == 0
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["artifacts"]["inventory"] == [{"kind": "manifest", "path": "x.json"}]


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

    assert runner_main([str(run_dir)]) == 0
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["lifecycle"]["inference"] == "failed"
    assert "was not created" in metadata["completion_error"]


def test_tmux_inside_links_adjacent_without_renaming_source(monkeypatch):
    calls = []
    tmux = Tmux("fake-tmux")

    class Result:
        def __init__(self, stdout=""):
            self.stdout = stdout
            self.returncode = 0

    def fake_run(args, **kwargs):
        calls.append(args)
        if "display-message" in args:
            return Result("4\n")
        return Result()

    monkeypatch.setattr("lasagna.manager.tmux.subprocess.run", fake_run)
    tmux.attach("las-example", environ={"TMUX": "yes"})
    assert ["fake-tmux", "link-window", "-a", "-s", "las-example:0", "-t", "4"] in calls
    assert not any("rename-window" in call for call in calls)
