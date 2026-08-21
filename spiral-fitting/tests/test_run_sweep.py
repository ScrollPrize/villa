import io
import sys
import threading

from runners import run_sweep


def test_relay_logs_everything_and_echoes_only_live_fit_updates():
    source = io.StringIO(
        "loading configuration\n"
        "PROGRESS Optimizing — 200/1,000 iterations (20.0%) — 4.0 it/s\n"
        "step 200: loss = 12.5, patch = 3.0\n"
        "rendering output\n"
    )
    log = io.StringIO()
    console = io.StringIO()

    run_sweep._relay_child_output(
        source, log, "baseline", console, threading.Lock())

    assert log.getvalue() == (
        "loading configuration\n"
        "PROGRESS Optimizing — 200/1,000 iterations (20.0%) — 4.0 it/s\n"
        "step 200: loss = 12.5, patch = 3.0\n"
        "rendering output\n"
    )
    assert console.getvalue() == (
        "[baseline] PROGRESS Optimizing — 200/1,000 iterations (20.0%) — 4.0 it/s\n"
        "[baseline] step 200: loss = 12.5, patch = 3.0\n"
    )


def test_started_child_is_unbuffered_and_final_output_is_drained(tmp_path):
    log_path = tmp_path / "child.log"
    console = io.StringIO()
    command = [
        sys.executable,
        "-c",
        "import os; "
        "print('PROGRESS Optimizing — 1/2 iterations'); "
        "print('step 200: loss = 1.0'); "
        "print('unbuffered=' + os.environ.get('PYTHONUNBUFFERED', ''))",
    ]

    with log_path.open("a") as log:
        process, relay = run_sweep._start_child(
            command, log, "trial", console, threading.Lock())
        assert process.wait(timeout=10) == 0
        relay.join(timeout=10)
        assert not relay.is_alive()

    assert log_path.read_text().splitlines() == [
        "PROGRESS Optimizing — 1/2 iterations",
        "step 200: loss = 1.0",
        "unbuffered=1",
    ]
    assert console.getvalue().splitlines() == [
        "[trial] PROGRESS Optimizing — 1/2 iterations",
        "[trial] step 200: loss = 1.0",
    ]
