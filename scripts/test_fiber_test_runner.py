import json
from types import SimpleNamespace

import pytest

import fiber_test_runner as runner


@pytest.mark.parametrize("config", [
    {}, [], {"suites": []}, {"suites": "manager-tests"},
    {"suites": ["unknown"]}, {"suites": [42]},
    {"suites": ["runner-tests", "runner-tests"]},
    {"suites": ["runner-tests"], "command": "echo forbidden"},
])
def test_reject_invalid_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner.load_suites(path)


def test_default_and_explicit_selection(tmp_path):
    path = tmp_path / "config.json"
    assert runner.load_suites(path) == ["runner-tests"]
    path.write_text('{"suites":["manager-tests","loader-tests"]}')
    assert runner.load_suites(path) == ["manager-tests", "loader-tests"]


def test_no_cli_commands():
    assert runner.main(["arbitrary-command"]) == 2


def test_missing_harness_fails_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(ValueError, match="Missing harness"):
        runner.run(["manager-tests"], tmp_path)


def test_exact_command_environment_and_fail_fast(monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=3)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unexpected")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.run(["manager-tests", "runner-tests"]) == 3
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [runner.sys.executable, "-m", "pytest", "-x", "-s", "-q",
                       "gpu_manager/tests/test_exclusive.py"]
    assert kwargs["cwd"] == runner.ROOT
    assert kwargs["env"]["PYTHONPATH"].split(runner.os.pathsep)[0] == str(runner.ROOT / "gpu_manager/src")
    assert "PYTEST_ADDOPTS" not in kwargs["env"]
