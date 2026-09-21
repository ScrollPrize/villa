# Using `vesuvius` on Google Colab

This guide covers running the `vesuvius` package on a Google Colab GPU session from the command line, instead of working inside the notebook UI. It is based on a working setup using the [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) tool, which drives a Colab runtime as a remote shell.

If you'd rather stay inside the browser notebook UI, you can also attach a Colab runtime through the "Colab" extension for VS Code, which lets you pick a CPU, GPU, or TPU runtime directly from the editor. The steps below assume the CLI-driven workflow, which is more reproducible and scriptable.


## 1. Install the Colab CLI

On your local machine:

```bash
uv tool install google-colab-cli
```

This gives you the `colab` command, authenticated against your Google account.

> **Known issue (google-colab-cli 0.6.0, the latest release as of writing):** it depends on `jupyter-kernel-client` unpinned, which recently crossed into a `1.0.x` release that renamed the class the CLI expects (`KernelClient` → `JupyterKernelClient`). This breaks `colab exec`, `colab repl`, `colab drivemount`, and `colab install` with `AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'` — confirmed against a live session. Work around it until upstream pins a compatible version:
> ```bash
> uv tool install google-colab-cli --with "jupyter-kernel-client<1.0.0" --force
> ```

## 2. Start a Colab session with GPU

```bash
colab new -s vesuvius --gpu t4
```

- `-s vesuvius` names the session so you can reconnect to it later.
- `--gpu` accepts `t4`, `l4`, or `a100`, in increasing order of cost/performance. `t4` is the cheapest and is enough for exploring the examples; use `l4`/`a100` for heavier training runs.

List your active sessions at any time with:

```bash
colab sessions
```

Reconnect to a running session with:

```bash
colab console -s vesuvius
```

### Persistence of Colab sessions

A Colab session/runtime is stopped (VM reclaimed, disk wiped, GPU released) when any of these hit:

1. Idle timeout — no active execution and no browser/CLI interaction for ~90 minutes. This is the most common trigger.
2. Max runtime duration — a hard ceiling regardless of activity:
   - Free tier: ~12 hours
   - Colab Pro: ~24 hours
   - Colab Pro+: can run longer and survive some backgrounding, but still not indefinite
3. You manually disconnect/terminate it — closing it explicitly (in the UI, or via colab CLI if it exposes a stop/delete command), or restarting the runtime.
4. Google reclaims the resource — if you're on free tier and Google needs the GPU capacity elsewhere, or you've hit usage limits for the day/week, it can be stopped even before the idle/max-duration limits.
5. Browser tab closed for too long (web UI specifically) — the frontend disconnecting doesn't instantly stop the runtime, but if nothing reconnects within the idle window, it times out per #1.

There's no way to make a Colab runtime permanent — even Pro+ just extends the numbers above, it doesn't remove them. That's the whole reason the guide's Drive-mounting step exists: assume the VM disappears without warning and keep anything you care about (weights, datasets, the rclone config) off the local disk.

## 3. Set up `vesuvius` inside the session

Once you have a shell inside the Colab console (from `colab console -s vesuvius`):

```bash
git clone https://github.com/<your-user>/villa.git
cd villa
bash volume-cartographer/scripts/install_build_deps.sh
cd vesuvius
uv sync --extra models
```

The `models` extra pulls in `volume-cartographer`'s C++ Python bindings (there is no separate Colab-only extra anymore), so `install_build_deps.sh` has to run first to install its system build dependencies (Qt6, OpenCV, CGAL, Ceres, etc.) — see "Known issues" below for the LLVM/CMake/GCC compatibility fixes already baked into that script for older Ubuntu releases.

Verify the GPU is visible to PyTorch:

```bash
uv run --extra models python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
```

## 4. Running scripts directly against a session (shebang execution)

`google-colab-cli` supports [shebang execution](https://github.com/googlecolab/google-colab-cli#shebang-execution-support), which lets a local script run transparently on a Colab GPU without an explicit `colab console` step. Add this as the first line of a script:

```python
#!/usr/bin/env -S colab run --gpu L4 --keep
import torch

print("L4 GPU Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))
```

Make it executable and run it like any local script:

```bash
chmod +x script.py
./script.py
```

`--keep` tells the CLI to preserve the session VM after the script finishes, so you can re-execute it or inspect logs without paying the cold-start cost again.

## 5. Mounting Google Drive for persistent storage

Colab session disks are ephemeral, so datasets and checkpoints you want to keep should live on Google Drive, mounted via `rclone`.

### First-time setup

Inside the Colab session:

```bash
apt install rclone
```

`rclone` needs its own Google API OAuth client (the default shared client is heavily rate-limited). Follow rclone's ["making your own client ID"](https://rclone.org/drive/#making-your-own-client-id) instructions to create one in the Google Cloud Console, then run:

```bash
rclone config
```

When prompted:
- **Name the remote `gdrive`** — later steps assume this exact name.
- Choose `drive` as the storage type and paste in your own client ID / secret.
- Do **not** try to supply a service-account JSON file or use automatic/browser-based config inside the Colab console — it doesn't have a way to open a browser. Use the "no" / manual/headless authorization flow when prompted, and it will give you a URL + code to authorize from a browser on your own machine.

Then mount the drive:

```bash
mkdir /content/drive
rclone mount gdrive: /content/drive --vfs-cache-mode writes &
```

### Making the rclone config reproducible across sessions

Since each new Colab session starts from a clean VM, redoing the `rclone config` wizard every time is painful. Which persistence method applies depends on whether you're driving Colab through the CLI (as the rest of this guide does) or through the notebook UI — they are **not** interchangeable: `google.colab.userdata` is a notebook-frontend API and is unreachable from a bare `colab console`/`colab exec` shell session, so it doesn't work for the CLI-driven workflow this guide otherwise uses.

#### CLI-native: keep the file locally, push it with `colab upload`

`colab upload`/`colab download`'s REMOTE path is a literal string handed to a file-transfer API — it is **not** shell-expanded on the remote side. Writing `~/...` there gets expanded by your *local* shell to *your own local* home directory before `colab` ever sees it — confirmed by testing: it silently tried to write to `/home/<local-user>/...` on the VM and failed with a 500 error. Resolve the session's real home directory first instead of guessing:

```bash
REMOTE_HOME=$(echo "import os; print(os.environ['HOME'])" | colab exec -s vesuvius --timeout 30)
```

(Currently `/root` for a `colab new`-provisioned session — but resolve it rather than hardcoding, in case that ever changes.)

1. After `rclone config` succeeds once inside a session, pull the resulting config down to your own machine so you never have to redo the wizard again:

   ```bash
   colab download -s vesuvius "$REMOTE_HOME/.config/rclone/rclone.conf" ./rclone.conf
   ```

   Keep `./rclone.conf` somewhere safe locally — it's a live credential (contains an OAuth token), not something to commit to the repo or leave lying around in a shared location.

2. At the start of every future session, push it onto the fresh VM before running any `rclone` command:

   ```bash
   echo "import pathlib; pathlib.Path('$REMOTE_HOME/.config/rclone').mkdir(parents=True, exist_ok=True)" | colab exec -s vesuvius --timeout 30
   colab upload -s vesuvius ./rclone.conf "$REMOTE_HOME/.config/rclone/rclone.conf"
   ```

   After that, `rclone mount gdrive: /content/drive --vfs-cache-mode writes &` just works — no wizard, no browser step, and no notebook kernel required.

> **Note on `colab exec` and stdin:** piping code in is just `echo "..." | colab exec -s NAME` — there's no `-f -` stdin marker; `-f FILE` always opens a real local file. To run a shell script (not Python) through `exec`, put `%%bash` as the first line of the file — since `exec` sends the file's content to the same Jupyter kernel a notebook cell would use, that line is treated as a cell magic and everything after it runs as bash, with output streamed back normally. Also note `exec` defaults to a 30-second execution timeout — pass `--timeout` with a much larger value for anything slower than that (e.g. `uv sync`, `apt-get install`, `git clone`).

#### Notebook UI alternative: Colab Secrets

If you're instead working inside the notebook UI rather than this guide's CLI-driven console workflow, you can persist the config as a Colab Secret instead:

1. Grab your working config from inside a session where `rclone config` already succeeded:

   ```bash
   cat ~/.config/rclone/rclone.conf
   ```

   Copy the whole output (the `[gdrive]` section, including the `token = {...}` line).

2. Store it as a Colab Secret (this persists across sessions, tied to your Google account, not the ephemeral VM):
   - Click the key icon in the Colab left sidebar.
   - Add a new secret named `RCLONE_CONF`.
   - Paste the full file content as the value.
   - Toggle "Notebook access" on.

3. At the start of every future notebook session, restore it before running any `rclone` command (this only works in an actual notebook cell, not a `colab console`/`colab exec` shell):

   ```python
   from google.colab import userdata
   import pathlib

   conf_dir = pathlib.Path.home() / ".config" / "rclone"
   conf_dir.mkdir(parents=True, exist_ok=True)
   (conf_dir / "rclone.conf").write_text(userdata.get('RCLONE_CONF'))
   ```

## 6. Downloading datasets from Hugging Face

To pull a dataset directly onto the mounted Drive (so it survives past the session):

```bash
uvx --from huggingface_hub hf buckets sync \
      hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002 \
      /content/drive/ink-dataset/phercparis4/w00_20231016151002
```

### Verifying the sync actually completed

`hf buckets sync` supports `--dry-run`: re-run the exact same command with it appended and it diffs source vs. destination (by size + mtime) and prints the sync plan as JSONL, without transferring anything.

```bash
uvx --from huggingface_hub hf buckets sync \
      hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002 \
      /content/drive/ink-dataset/phercparis4/w00_20231016151002 \
      --dry-run
```

- **Empty output** — the destination already matches the bucket exactly; the original sync fully succeeded.
- **Files listed** — those are missing/mismatched; the original run was incomplete or failed partway. Drop `--dry-run` to fetch just the gap.

Two more things worth checking:

- **Exit code**: if you're still in the session that ran the sync, `echo $?` right after the command tells you pass/fail directly. If that session is gone, the `--dry-run` diff above is the only retroactive signal you have.
- **VFS cache vs. actually-on-Drive**: the `rclone mount ... --vfs-cache-mode writes` setup from step 5 caches writes locally before flushing them up to Drive. Files can look present under `/content/drive/...` while still sitting in local cache, not yet durably stored in your Google Drive — if the Colab VM got reclaimed before that flush finished, the data could be gone even though the sync looked done. Confirm it actually landed by querying the remote directly (bypassing the local cache):

  ```bash
  rclone size gdrive:ink-dataset/phercparis4/w00_20231016151002
  ```

  or just check the folder in the Google Drive web UI — that's ground truth regardless of whether the Colab session that ran the sync still exists.

## Troubleshooting: console appears stuck during installs

`colab console -s <name>` proxies a shell over a network connection, and that connection can stall or silently drop while a long install (`install_build_deps.sh` compiling Qt6/OpenCV/CGAL/Ceres, or `uv sync --extra models` pulling `torch`, `cucim-cu13`, `nnunetv2`, and their dependency trees) is still running in the foreground. When that happens the console looks frozen, and if the connection actually re-attaches, whatever was running in the plain foreground shell is killed along with it.

Avoid this by never running installs directly in the bare console shell — run them inside `screen` (or `tmux`) so the process is detached from the console connection itself:

```bash
sudo apt install screen   # once per session, if not already present
screen -S setup
uv sync --extra models
```

If the console appears to hang, detach with `Ctrl-A` then `D` rather than killing the terminal — the install keeps running on the VM. Reconnect and resume watching it with:

```bash
colab console -s vesuvius
screen -r setup
```

## Known issues

- **`volume-cartographer` install/compile failures on older Ubuntu (e.g. Jammy 22.04, which both local dev machines and Colab may run).** `volume-cartographer/scripts/install_build_deps.sh` originally assumed packages only available on the newer Ubuntu release CI builds against (`ubuntu:26.04`): LLVM 21 (`flang-21`/`libclang-rt-21-dev`), CMake ≥3.28, and a GCC new enough for C++23's `std::byteswap` (GCC ≥13). The script now detects and works around all three automatically (adding `apt.llvm.org` and `ubuntu-toolchain-r/test` as needed, and `vesuvius/pyproject.toml`'s `cmake.version` constraint was corrected to match `CMakeLists.txt`'s real minimum) — these are no-ops wherever the OS already has new-enough versions. There is no longer a separate Colab-only extra: `models` includes `volume-cartographer` again, so step 3 above always needs `install_build_deps.sh` run first.
- **`colab drivemount` and Colab Secrets (`google.colab.userdata`) do not work from a CLI-driven session.** Confirmed against a live session: `drivemount`'s `google.colab.drive.mount()` call times out waiting on an OAuth handoff, and `userdata.get()` raises `google.colab.userdata.TimeoutException: ... Secrets can only be fetched when running from the Colab UI.` Both require a live Colab browser tab connected to the runtime, which a bare `colab console`/`colab exec` session never has — this is independent of the `jupyter-kernel-client` bug noted in step 1. Use the CLI-native `rclone` + `colab upload`/`colab download` approach in the Drive-mounting section instead.

If a command genuinely looks stuck rather than just slow, check whether it's still making progress (e.g. re-run with `uv sync -v`, or check network activity) before assuming it's hung — some of these downloads are large and Colab's outbound bandwidth is not always fast.

## Automating all of this: `scripts/colab_bootstrap.sh`

Everything in steps 1–3 and 5–6 above is automated in [`scripts/colab_bootstrap.sh`](../scripts/colab_bootstrap.sh), driven entirely from your local machine:

```bash
REPO_URL=https://github.com/<your-user>/villa.git ./scripts/colab_bootstrap.sh
```

It applies the `jupyter-kernel-client` compatibility pin if needed, creates or reuses the named session, resolves the session's real `$HOME` (never hardcode `/root` — see the CLI-native rclone section above for why), pushes an existing local `rclone.conf` and mounts Drive if `RCLONE_CONF_LOCAL` points at one, clones/updates the repo (pass `REPO_BRANCH` to target a branch other than the default), checks a Drive-backed wheel cache for `volume-cartographer` before falling back to `install_build_deps.sh` + a from-source build with a RAM-aware parallelism cap (see below), runs `uv sync --extra models`, and verifies GPU visibility. Hugging Face dataset syncing was deliberately left out of this script — it doesn't need a GPU, isn't really part of environment setup, and is arguably better run somewhere with more persistent storage and bandwidth than an ephemeral Colab VM (see the manual `hf buckets sync` steps above instead). See the script's header comment for the full list of environment variables and the one-time manual prerequisites it can't do for you (creating your own rclone OAuth client and completing `rclone config` once).

### Build-time results (2026-09-20/21)

| GPU tier | vCPUs (`nproc`) | Job count | Total wall time | Notes |
| --- | --- | --- | --- | --- |
| T4 | 2  | 2 (RAM-aware cap)       | **12m45s** | From-source compile, full pipeline, clean run |
| T4 | 2  | ~4 (Ninja default, uncapped) | **16m50s** | From-source compile, full pipeline, clean run — **slower** than capped |
| L4 | 12 | 12 (RAM-aware cap)      | **6m09s**  | From-source compile, full pipeline, clean run |
| L4 | 12 | ~14 (Ninja default, uncapped) | **~6 min** (partial) | From-source compile, build step only — see caveat below |
| T4 | — | n/a (cache hit) | **~3m31s** | Wheel cache hit, full pipeline including GPU verify, clean run |
| L4 | — | n/a (cache hit) | **~3m51s** | Wheel cache hit, full pipeline including GPU verify, clean run |

The T4 capped-vs-uncapped comparison is the clean, trustworthy from-source result: capping build parallelism to match actual core count (rather than trusting Ninja's `nproc+2` default, which oversubscribes a 2-core VM) made the build **faster**, not just safer — 12m45s vs 16m50s.

The L4 uncapped-from-source number is **not a clean measurement**: getting a full, clean from-scratch run on an uncapped L4 session failed four consecutive times for unrelated infrastructure reasons (see "`colab exec` reliability" below), and the ~6 minute figure is only the build step in isolation. It's included because it's suggestive — close to the capped L4 total, meaning the cap likely barely constrains L4 at all — but treat it as a data point to redo, not a confirmed result.

The two **cache-hit** rows are the real payoff of the wheel-caching work below: both are clean, fully confirmed runs (`torch 2.12.1+cu130 | cuda: True` on both), and both land around **3.5–4 minutes regardless of GPU tier** — because a cache hit skips the compile entirely, the GPU tier's core count stops mattering; the bottleneck becomes package download/install speed, which is roughly constant. That's a **~3.5–4x speedup on T4** and a **~1.6-1.8x speedup on L4** versus a from-source build on the same tier.

### Speeding up repeated builds: caching a prebuilt wheel

**Implemented and confirmed working** in `colab_bootstrap.sh`:

- **Cache key**: the git tree hash of `volume-cartographer/` specifically (`git rev-parse HEAD:volume-cartographer`), not the whole repo's commit — so unrelated changes elsewhere in the monorepo don't invalidate a perfectly good cached wheel, but any real change under `volume-cartographer/` does.
- **Cache location**: `/content/drive/vesuvius/vc-wheel-cache/<tree-hash>/*.whl` on the mounted Drive — only active when `RCLONE_CONF_LOCAL` is set; otherwise falls back to a session-local `/root/.cache/vc-wheel` with no cross-session reuse.
- **Portability**: confirmed safe. `VC_MARCH_NATIVE` defaults to `OFF` in `CMakeLists.txt`, and the non-native build path compiles for the portable `-march=x86-64-v3` baseline, not `-march=native` — so a wheel built on one Colab host's CPU runs fine on another, and the same cached wheel was successfully reused across both a T4 and an L4 session.
- **Mechanics**: `uv sync --extra models --no-install-package volume-cartographer` always runs first (fast, resolves everything else from the lockfile). On a cache hit, the cached wheel is installed directly; on a miss, `install_build_deps.sh` runs, the RAM-aware job cap is computed, `uv build --wheel` compiles exactly once, and the result is both installed and left in the cache dir for next time.
- **Known limitation**: `install_build_deps.sh` still runs on every cache miss (needed for the runtime shared-library deps the compiled `.so` links against — Qt6/OpenCV/CGAL/Ceres — not just for building), so a cache hit doesn't skip *all* setup work, only the actual C++ compile.

Three non-obvious bugs surfaced while implementing and validating this, all now fixed in `colab_bootstrap.sh`:

- **`uv pip install <wheel>` does not auto-detect the project's `.venv`** the way `uv sync`/`uv run` do — a bare call resolved to the VM's *system* Python (3.13) instead of the project's 3.14 venv, and installing a `cp314`-tagged wheel into a 3.13 environment fails outright. Fixed by passing `--python .venv/bin/python` explicitly on every `uv pip install` call.
- **`uv run` re-syncs the environment against `uv.lock` before running, even with `--no-install-package` used earlier in the flow.** Since `volume-cartographer` is declared as an editable path dependency, a plain `uv run` after a cache-hit wheel install tries to *rebuild it as editable*, undoing the cache hit and failing outright (since `install_build_deps.sh` was skipped, so `Ceres`/etc. aren't present to build against). Fixed with `uv run --no-sync` for the GPU-verification step.
- **rclone's Drive VFS mount can be too slow for a 30s timeout** on a simple `ls` of a path that doesn't exist yet (the very first cache check, before anything has ever been cached) — this isn't a hang, just real Drive API latency. Bumped that specific check to 90s.

## `colab exec` reliability: client-side hangs and session death

While gathering the build-time numbers above, `colab exec` calls hung or the underlying session died outright on multiple separate occasions, independent of anything in this repo's build scripts:

- **Client-side hangs that `--timeout` doesn't catch.** `colab exec --timeout N` only bounds execution time *inside the remote kernel* — it does nothing if the client's connection to the backend itself hangs. Observed directly, twice: a call sat blocked for over 2 hours in one case, and ~51 minutes in another, while `colab status` showed the session as `IDLE` and its own execution log confirmed the remote code had actually finished within a few minutes. The client was simply never woken up by the response. `colab_bootstrap.sh` now wraps every `colab exec` call in a hard client-side `timeout` (declared timeout + 60s grace) so a hung connection fails loudly within a bounded time instead of stalling the whole script indefinitely — but this is a workaround, not a fix for the underlying transport bug.
- **Sessions can die (404/401) mid-run with no warning**, including immediately after a step completed successfully (observed right after a `uv build --wheel` finished cleanly). `colab status`/`colab sessions` local tracking can also silently drop an entry (showing `[?]` or "not found") for a session that may still be alive and billed/quota'd server-side — the CLI's local bookkeeping isn't fully reliable for this.
- **Concurrent-session quota**: creating several sessions in a short span (we had ~5–6 active from this benchmarking work) triggered `TooManyAssignmentsError` (HTTP 412) on a new `colab new`. If local `colab sessions` doesn't show a session you expect to still be running, check the Colab web UI directly rather than trusting the CLI's view — it can drift from what's actually alive server-side.
- **`colab exec` also has a genuine remote-side (not just client-side) timeout mode**: `jupyter_kernel_client`'s own wait loop can raise `TimeoutError: Timeout waiting for output` after the declared `--timeout` elapses, printed as a normal (non-hanging) CLI error. This happened on an otherwise-fast cache-hit install for no obvious reason — the session was `IDLE` with the work already done seconds later. Treat it the same way as a client-side hang: check `colab status -s <name>` before assuming the step didn't happen, and resume rather than restart.
- **A bash scripting bug in `colab_bootstrap.sh` itself was silently swallowing exactly these errors** for most of this benchmarking session, before being found and fixed: under `set -euo pipefail`, a bare assignment like `out=$(cmd)` kills the *entire script* immediately if `cmd` returns nonzero — before a following `rc=$?` line, or any error-printing code, ever runs. Since `colab exec` itself exits nonzero on real CLI-level failures (a lost session, a genuine remote timeout) — as opposed to a remote Python exception, which it prints but exits 0 for — every one of those CLI-level failures was killing the script with **zero diagnostic output**, showing up only as a bare "exited with code 1" with nothing above it. This is very likely the actual explanation for several earlier "silent failure, no traceback" incidents in this same session that got attributed to hangs or session death without a clear mechanism. Fixed by using `out=$(cmd) || rc=$?` (which bash's `errexit` correctly treats as a checked failure, not a fatal one) everywhere a remote call's output is captured, plus a dedicated `check_exec_rc` that reports client-side hangs (124) and CLI-level failures (any other nonzero) with an explicit message before exiting — instead of relying only on scanning output content for a Python traceback, which says nothing when the CLI itself never got that far.

Practical upshot: for anything time-sensitive or unattended, don't assume a single `colab_bootstrap.sh` invocation will complete cleanly on the first try — check `colab status -s <name>` before concluding a hang means the work didn't happen, and be prepared to resume against the same (still-`IDLE`) session rather than starting over from scratch. And if you're writing your own wrapper around `colab exec`/any CLI like it under `set -e`, watch for exactly this assignment gotcha — it hides real errors as bare exit codes with no explanation.
