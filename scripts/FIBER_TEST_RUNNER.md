# Reusable fiber test and experiment runner

Approve this command once and reuse the identical command:

```sh
/home/hendrik/villa3/scripts/run_fiber_tests
```

It takes no CLI arguments. Select named suites in the untracked
`local/fiber_tests.json`, for example:

```json
{"suites": ["runner-tests", "manager-tests"]}
```

Without a config it runs only the launcher unit tests. The runner prints the
configuration and selected tests, runs sequentially, and stops on the first
failure. Unknown keys/suites, duplicate suites and missing harnesses fail before
execution. There are no arbitrary shell commands or pytest argument passthrough.
No installs, commits, production restarts or direct GPU allocation are performed.

| Suite | Work |
|---|---|
| `runner-tests` | Launcher validation/unit tests; CPU only |
| `manager-tests` | Manager command/partial-grant/cleanup regressions; synthetic GPU inventory, temporary local sockets |
| `loader-build` | Existing isolated queued-wake CMake configure/build harness |
| `loader-tests` | Native cache/queue and related regression harnesses; may build their test targets |
| `loader-scaling` | Existing matched 1/16/64-worker experiment; requests its GPU/CPU grant through the manager |
| `loader-report` | Recreate the scaling report from saved measurements |

Loader suites require the existing `local/test_queued_*` harnesses and build/data
fixtures in this workspace; they fail explicitly on fresh clones without those
fixtures. Scaling retains the harness's resume behavior: existing completed
measurements are reused, not silently retimed. Change the experiment's output
directory when fresh timings are intended. New experiments should add a named
suite, not an arbitrary command option. Announce changed suite selection before
running it; approval of this launcher does not authorize unrelated work.

The launcher uses the existing `/home/hendrik/.venv_las/bin/python` environment.
On other hosts set `VILLA_TEST_PYTHON` to an existing interpreter with pytest.
It sets repository imports and single-thread BLAS defaults, and never installs
an environment. Long GPU work stays under the existing manager-aware harness.
