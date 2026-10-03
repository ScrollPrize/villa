# Using `vesuvius` on Google Colab

This guide covers running the `vesuvius` package on a Google Colab GPU session from the command line, instead of working inside the notebook UI. It is based on a working setup using the [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) tool, which drives a Colab runtime as a remote shell.

If you'd rather stay inside the browser notebook UI, you can also attach a Colab runtime through the "Colab" extension for VS Code, which lets you pick a CPU, GPU, or TPU runtime directly from the editor. The steps below assume the CLI-driven workflow, which is more reproducible and scriptable.

## TL;DR

Most of this guide is automated by a few scripts. From your local machine, once you have your own rclone Drive OAuth client set up (one-time — see "Mounting Google Drive" below) and `RCLONE_CONF_LOCAL` pointed at a working `rclone.conf`:

```bash
# One-time: install the CLI
uv tool install google-colab-cli

# Set up a Colab GPU session: creates/reuses it, mounts Drive, clones the
# repo, installs everything (fast if the volume-cartographer wheel cache hits)
REPO_URL=https://github.com/<you>/villa.git \
RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
./scripts/colab_bootstrap.sh

# Run a training job: reuses the setup above, uploads a config onto the
# mounted Drive, and launches training detached so it survives disconnects
REPO_URL=https://github.com/<you>/villa.git \
RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
CONFIG_LOCAL=../configs/ink_tutorial.json \
./scripts/colab_train.sh

# Run inference with a trained checkpoint; the prediction TIFF lands on Drive
REPO_URL=https://github.com/<you>/villa.git \
RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
CHECKPOINT_REMOTE=/content/drive/vesuvius/runs/ink_tutorial_s3/ckpt_020000.pth \
./scripts/colab_infer.sh
```

- Training data can come from a Drive-synced dataset (see "6. Downloading datasets" below) or, for datasets already published to the Vesuvius Challenge S3 bucket, directly from `s3://` with no local copy at all (see "7. Reading input data" below) — either way, training *outputs* (checkpoints, logs, the uploaded config) still land on Drive.
- Full details on each script, every environment variable, and real build-time/throughput numbers are in "Quick start: automation scripts" just below.
- The numbered steps further down (1–7) are the manual walkthrough these scripts automate — read them if you want to understand what's happening under the hood, do it by hand, or debug something the scripts don't surface clearly.

## Quick start: automation scripts

### `scripts/colab_bootstrap.sh`: environment setup

Everything in the manual steps 1–3 and 5–6 below is automated in [`scripts/colab_bootstrap.sh`](../scripts/colab_bootstrap.sh), driven entirely from your local machine:

```bash
REPO_URL=https://github.com/<your-user>/villa.git ./scripts/colab_bootstrap.sh
```

It applies the `jupyter-kernel-client` compatibility pin if needed, creates or reuses the named session, resolves the session's real `$HOME` (never hardcode `/root` — see the CLI-native rclone section in step 5 below for why), pushes an existing local `rclone.conf` and mounts Drive if `RCLONE_CONF_LOCAL` points at one, clones/updates the repo (pass `REPO_BRANCH` to target a branch other than the default), checks a Drive-backed wheel cache for `volume-cartographer` before falling back to `install_build_deps.sh` + a from-source build with a RAM-aware parallelism cap (see below), runs `uv sync --extra models`, and verifies GPU visibility. Hugging Face dataset syncing was deliberately left out of this script — it doesn't need a GPU, isn't really part of environment setup, and is arguably better run somewhere with more persistent storage and bandwidth than an ephemeral Colab VM (see the manual `hf buckets sync` steps in step 6 below instead). See the script's header comment for the full list of environment variables and the one-time manual prerequisites it can't do for you (creating your own rclone OAuth client and completing `rclone config` once).

#### Build-time results (2026-09-20/21)

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

#### Speeding up repeated builds: caching a prebuilt wheel

**Implemented and confirmed working** in `colab_bootstrap.sh`:

- **Cache key**: the git tree hash of `volume-cartographer/` specifically (`git rev-parse HEAD:volume-cartographer`), not the whole repo's commit — so unrelated changes elsewhere in the monorepo don't invalidate a perfectly good cached wheel, but any real change under `volume-cartographer/` does.
- **Cache location**: `/content/drive/vesuvius/vc-wheel-cache/<tree-hash>/*.whl` on the mounted Drive — only active when `RCLONE_CONF_LOCAL` is set; otherwise falls back to a session-local `/root/.cache/vc-wheel` with no cross-session reuse. The benchmark runs below originally used a copy of the shared/default rclone client (rate-limited, see the HF sync troubleshooting note in step 6 below); re-running against a freshly-reconnected custom-client credential confirmed the exact same cache (and cached wheel) is visible either way, since both point at the same underlying Drive account. Which credential to use is a tradeoff, not a solved problem — see the "token expired" troubleshooting note in step 5 below: the custom client avoids rate limits but needs reconnecting roughly every 7 days, and the shared client needs no maintenance but is rate-limited under bulk load.
- **Portability**: confirmed safe. `VC_MARCH_NATIVE` defaults to `OFF` in `CMakeLists.txt`, and the non-native build path compiles for the portable `-march=x86-64-v3` baseline, not `-march=native` — so a wheel built on one Colab host's CPU runs fine on another, and the same cached wheel was successfully reused across both a T4 and an L4 session.
- **Mechanics**: `uv sync --extra models --no-install-package volume-cartographer` always runs first (fast, resolves everything else from the lockfile). On a cache hit, the cached wheel is installed directly; on a miss, `install_build_deps.sh` runs, the RAM-aware job cap is computed, `uv build --wheel` compiles exactly once, and the result is both installed and left in the cache dir for next time.
- **Known limitation**: `install_build_deps.sh` still runs on every cache miss (needed for the runtime shared-library deps the compiled `.so` links against — Qt6/OpenCV/CGAL/Ceres — not just for building), so a cache hit doesn't skip *all* setup work, only the actual C++ compile.

Three non-obvious bugs surfaced while implementing and validating this, all now fixed in `colab_bootstrap.sh`:

- **`uv pip install <wheel>` does not auto-detect the project's `.venv`** the way `uv sync`/`uv run` do — a bare call resolved to the VM's *system* Python (3.13) instead of the project's 3.14 venv, and installing a `cp314`-tagged wheel into a 3.13 environment fails outright. Fixed by passing `--python .venv/bin/python` explicitly on every `uv pip install` call.
- **`uv run` re-syncs the environment against `uv.lock` before running, even with `--no-install-package` used earlier in the flow.** Since `volume-cartographer` is declared as an editable path dependency, a plain `uv run` after a cache-hit wheel install tries to *rebuild it as editable*, undoing the cache hit and failing outright (since `install_build_deps.sh` was skipped, so `Ceres`/etc. aren't present to build against). Fixed with `uv run --no-sync` for the GPU-verification step.
- **rclone's Drive VFS mount can be too slow for a 30s timeout** on a simple `ls` of a path that doesn't exist yet (the very first cache check, before anything has ever been cached) — this isn't a hang, just real Drive API latency. Bumped that specific check to 90s.

### `scripts/colab_train.sh`: launching a training run

[`scripts/colab_train.sh`](../scripts/colab_train.sh) builds on `colab_bootstrap.sh` for a full, single-command training launch: it runs bootstrap for environment setup (session, Drive mount, `vesuvius` install), uploads a local training config onto the mounted Drive, then launches training detached on the remote session — so a multi-hour run survives client disconnects and the `colab exec` reliability issues documented below.

```bash
REPO_URL=https://github.com/<you>/villa.git \
RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
./scripts/colab_train.sh
```

Unlike `colab_bootstrap.sh`, `RCLONE_CONF_LOCAL` is **required** here, not optional — training needs somewhere durable to write checkpoints, logs, and the config itself, and that's Drive regardless of where the training *input* data comes from (Drive-synced or read live from S3, see step 7 below).

Environment variables (all optional except `REPO_URL` and `RCLONE_CONF_LOCAL`; every `colab_bootstrap.sh` variable — `REPO_BRANCH`, `VESUVIUS_COLAB_SESSION`, `VESUVIUS_COLAB_GPU`, etc. — is also passed through):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONFIG_LOCAL` | `../configs/ink_tutorial.json` | Local training config to upload |
| `CONFIG_REMOTE` | `/content/drive/vesuvius/configs/<config-name>` | Where it lands on the mounted Drive |
| `TRAIN_MODULE` | `vesuvius.ink_detection.training.train` | Python module to run |
| `TRAIN_LOG_REMOTE` | `/content/drive/vesuvius/runs/<config-name>.log` | Training log path on Drive |
| `WANDB_API_KEY` | *(unset)* | Only needed if the config sets `wandb_project`; passed by reference at invocation time (`export WANDB_API_KEY=...` on the remote), never embedded in the script or written to disk anywhere |

Credentials are never embedded in the script: `RCLONE_CONF_LOCAL` and `WANDB_API_KEY` are both paths/values passed through at invocation time, handled exactly the way `colab_bootstrap.sh` already handles `RCLONE_CONF_LOCAL`.

Once launched, the script prints the training PID and the first ~30 lines of log output, then returns control to you — training keeps running on the VM (`nohup ... & disown`) after the script (and your terminal) exits. To check on it later:

```bash
# Tail the log directly
colab exec -s vesuvius --timeout 30 <<< "import subprocess; print(subprocess.run(['tail','-n','50','/content/drive/vesuvius/runs/ink_tutorial.log'],capture_output=True,text=True).stdout)"

# Or just reconnect and watch it live
colab console -s vesuvius
```

Checkpoints and previews land under the config's `out_dir` on Drive as training progresses, so they persist even if the session is later reclaimed — see the "Persistence of Colab sessions" note in step 2 below for why that matters.

There is no separate "final model" file: the last `ckpt_<num_iterations>.pth` *is* the trained model, and inference loads it directly. A `best_<metric>.pth` (plus `best_checkpoint.json`) is only written when the config sets `best_checkpoint_metric` (`val_loss` or `val_balanced_accuracy`); without it you only get the periodic `ckpt_*.pth` files.

### `scripts/colab_infer.sh`: running inference with a trained checkpoint

[`scripts/colab_infer.sh`](../scripts/colab_infer.sh) runs flat ink-detection inference (`vesuvius.ink_detection.inference.infer`) on a Colab GPU session. It reuses `colab_bootstrap.sh` for setup, launches inference detached, writes the prediction TIFF to local VM disk and copies it onto Drive when finished, then (by default) waits for completion and stops the session so it doesn't keep holding a GPU slot.

```bash
REPO_URL=https://github.com/<you>/villa.git \
REPO_BRANCH=<branch with your ink_detection changes> \
RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
CHECKPOINT_REMOTE=/content/drive/vesuvius/runs/ink_tutorial_s3/ckpt_020000.pth \
./scripts/colab_infer.sh
```

The model, normalization and input geometry are rebuilt from the config embedded in the checkpoint, so no training config is required. The input surface volume defaults to the first `surface_volume_paths` entry of `CONFIG_LOCAL` (`../configs/ink_tutorial_s3.json`), i.e. the segment the checkpoint was trained on; `s3://` URLs are read directly.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHECKPOINT_REMOTE` | *(required)* | Checkpoint path on the mounted Drive |
| `CONFIG_LOCAL` | `../configs/ink_tutorial_s3.json` | Config to read the default `INPUT_ZARR` from |
| `INPUT_ZARR` | first `surface_volume_paths` entry | Surface volume zarr (local path or `s3://` URL) |
| `OUTPUT_NAME` | `<run-dir>_<checkpoint-name>` | Basename of the outputs |
| `OUTPUT_DIR_REMOTE` | `/content/drive/vesuvius/predictions` | Drive folder for `<name>.tif`, `<name>.log`, `<name>.status` |
| `INFER_ARGS` | *(empty)* | Extra `infer` flags, e.g. `--tta-mirror --batch-size 4` |
| `WAIT` / `STOP_SESSION` | `1` / `1` | Wait for completion locally, then stop the session |
| `POLL_SECONDS` | `60` | Status-file polling interval |
| `STALE_MINUTES` | `10` | Declare a session dead after this long without a heartbeat |
| `RESUME_EVERY` | `500` | Blocks between resume snapshots (~4 min on a T4) |
| `KEEPALIVE_SECONDS` | `300` | Interval of a no-op `colab exec` while waiting (sessions were reclaimed ~20 min after the last exec); `0` disables |
| `RESUME_DIR_REMOTE` | `<OUTPUT_DIR_REMOTE>/<OUTPUT_NAME>.resume` | Durable resume state on Drive; deleted after success |
| `MAX_ATTEMPTS` | `40` | Fresh sessions to try before giving up |

Completion is detected from `<name>.status` on Drive (`ok`, or `failed (exit N)`), which is written only after the TIFF has been copied — the same "ground truth is a file on Drive, not the CLI's session status" approach as `colab_resume_watchdog.sh`. While it runs, `<name>.progress` on Drive is rewritten every minute with the log tail, RAM and GPU usage; a session counts as dead once that file stops changing for `STALE_MINUTES` (`colab status` was observed listing no sessions at all while one was healthy, so it isn't used).

**Inference is resumable, because a whole segment doesn't fit in one session.** Measured on a T4 (2026-10-03): a full PHercParis4 segment (`31960×51960`, 64 layers) is 85,750 patches at ~2.2 patches/s, i.e. ~9–10 hours, while free-tier sessions died after ~40 minutes — during both training and inference. So `colab_infer.sh` runs `infer` with `--resume-dir` on Drive: every `RESUME_EVERY` blocks it snapshots the completed output chunks, the still-open chunk buffers and the next block index (`vesuvius/ink_detection/inference/resume.py`). When a session dies, the script starts a fresh one (`<session>-r<N>`) and reruns the identical command, which continues from the last snapshot. The output is bit-identical to an uninterrupted run (same blocks, same accumulation order, buffers restored exactly — covered by `tests/ink_detection/test_flat_inference_resume.py`). Each attempt pays ~10 min of setup plus ~2 min of model load/compile/occupancy scan, so expect roughly two-thirds of each session to be useful inference. Resuming with a different input, checkpoint, geometry or batch size is refused instead of mixing results; use another `RESUME_DIR_REMOTE` or delete the old one.

Set `REPO_BRANCH` whenever your fork's default branch lacks your changes: `colab_bootstrap.sh` otherwise clones the default branch, whose `volume-cartographer` tree may not match the Drive wheel cache (forcing a from-source compile) or may no longer build on Colab's Ubuntu at all.

## `colab exec` reliability: client-side hangs and session death

While gathering the build-time numbers above, `colab exec` calls hung or the underlying session died outright on multiple separate occasions, independent of anything in this repo's build scripts:

- **Client-side hangs that `--timeout` doesn't catch.** `colab exec --timeout N` only bounds execution time *inside the remote kernel* — it does nothing if the client's connection to the backend itself hangs. Observed directly, twice: a call sat blocked for over 2 hours in one case, and ~51 minutes in another, while `colab status` showed the session as `IDLE` and its own execution log confirmed the remote code had actually finished within a few minutes. The client was simply never woken up by the response. `colab_bootstrap.sh` now wraps every `colab exec` call in a hard client-side `timeout` (declared timeout + 60s grace) so a hung connection fails loudly within a bounded time instead of stalling the whole script indefinitely — but this is a workaround, not a fix for the underlying transport bug.
- **Sessions can die (404/401) mid-run with no warning**, including immediately after a step completed successfully (observed right after a `uv build --wheel` finished cleanly). `colab status`/`colab sessions` local tracking can also silently drop an entry (showing `[?]` or "not found") for a session that may still be alive and billed/quota'd server-side — the CLI's local bookkeeping isn't fully reliable for this.
- **Concurrent-session quota**: creating several sessions in a short span (we had ~5–6 active from this benchmarking work) triggered `TooManyAssignmentsError` (HTTP 412) on a new `colab new`. If local `colab sessions` doesn't show a session you expect to still be running, check the Colab web UI directly rather than trusting the CLI's view — it can drift from what's actually alive server-side.
- **`colab exec` also has a genuine remote-side (not just client-side) timeout mode**: `jupyter_kernel_client`'s own wait loop can raise `TimeoutError: Timeout waiting for output` after the declared `--timeout` elapses, printed as a normal (non-hanging) CLI error. This happened on an otherwise-fast cache-hit install for no obvious reason — the session was `IDLE` with the work already done seconds later. Treat it the same way as a client-side hang: check `colab status -s <name>` before assuming the step didn't happen, and resume rather than restart.
- **A bash scripting bug in `colab_bootstrap.sh` itself was silently swallowing exactly these errors** for most of this benchmarking session, before being found and fixed: under `set -euo pipefail`, a bare assignment like `out=$(cmd)` kills the *entire script* immediately if `cmd` returns nonzero — before a following `rc=$?` line, or any error-printing code, ever runs. Since `colab exec` itself exits nonzero on real CLI-level failures (a lost session, a genuine remote timeout) — as opposed to a remote Python exception, which it prints but exits 0 for — every one of those CLI-level failures was killing the script with **zero diagnostic output**, showing up only as a bare "exited with code 1" with nothing above it. This is very likely the actual explanation for several earlier "silent failure, no traceback" incidents in this same session that got attributed to hangs or session death without a clear mechanism. Fixed by using `out=$(cmd) || rc=$?` (which bash's `errexit` correctly treats as a checked failure, not a fatal one) everywhere a remote call's output is captured, plus a dedicated `check_exec_rc` that reports client-side hangs (124) and CLI-level failures (any other nonzero) with an explicit message before exiting — instead of relying only on scanning output content for a Python traceback, which says nothing when the CLI itself never got that far.

Practical upshot: for anything time-sensitive or unattended, don't assume a single `colab_bootstrap.sh` invocation will complete cleanly on the first try — check `colab status -s <name>` before concluding a hang means the work didn't happen, and be prepared to resume against the same (still-`IDLE`) session rather than starting over from scratch. And if you're writing your own wrapper around `colab exec`/any CLI like it under `set -e`, watch for exactly this assignment gotcha — it hides real errors as bare exit codes with no explanation.

## The manual walkthrough

Everything below is what the two scripts above automate, spelled out step by step — useful for understanding what's actually happening, doing it by hand, or debugging something the scripts don't surface clearly.

### 1. Install the Colab CLI

On your local machine:

```bash
uv tool install google-colab-cli
```

This gives you the `colab` command, authenticated against your Google account.

> **Known issue (google-colab-cli 0.6.0, the latest release as of writing):** it depends on `jupyter-kernel-client` unpinned, which recently crossed into a `1.0.x` release that renamed the class the CLI expects (`KernelClient` → `JupyterKernelClient`). This breaks `colab exec`, `colab repl`, `colab drivemount`, and `colab install` with `AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'` — confirmed against a live session. Work around it until upstream pins a compatible version:
> ```bash
> uv tool install google-colab-cli --with "jupyter-kernel-client<1.0.0" --force
> ```

### 2. Start a Colab session with GPU

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

#### Persistence of Colab sessions

A Colab session/runtime is stopped (VM reclaimed, disk wiped, GPU released) when any of these hit:

1. Idle timeout — no active execution and no browser/CLI interaction for ~90 minutes. This is the most common trigger.
2. Max runtime duration — a hard ceiling regardless of activity:
   - Free tier: ~12 hours
   - Colab Pro: ~24 hours
   - Colab Pro+: can run longer and survive some backgrounding, but still not indefinite
3. You manually disconnect/terminate it — closing it explicitly (in the UI, or via colab CLI if it exposes a stop/delete command), or restarting the runtime.
4. Google reclaims the resource — if you're on free tier and Google needs the GPU capacity elsewhere, or you've hit usage limits for the day/week, it can be stopped even before the idle/max-duration limits.
5. Browser tab closed for too long (web UI specifically) — the frontend disconnecting doesn't instantly stop the runtime, but if nothing reconnects within the idle window, it times out per #1.

There's no way to make a Colab runtime permanent — even Pro+ just extends the numbers above, it doesn't remove them. That's the whole reason this guide's Drive-mounting step exists: assume the VM disappears without warning and keep anything you care about (weights, datasets, the rclone config) off the local disk.

### 3. Set up `vesuvius` inside the session

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

### 4. Running scripts directly against a session (shebang execution)

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

### 5. Mounting Google Drive for persistent storage

Colab session disks are ephemeral, so datasets and checkpoints you want to keep should live on Google Drive, mounted via `rclone`.

#### First-time setup

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

#### Troubleshooting: `oauth2: cannot fetch token: 401 Unauthorized` / `"unauthorized_client"` (token expired)

Expect to hit this **repeatedly, roughly every 7 days**, not just once. `drive`'s full-access scope is a Google "restricted" scope, and for any *unverified* app using a restricted scope, Google caps the refresh token to a 7-day lifetime — and this applies **regardless of whether the OAuth consent screen's publishing status is "Testing" or "In production."** Publishing to production does not remove this cap for a restricted scope; only completing Google's full app-verification process does (a real security-review process — domain/ownership verification, a security assessment, etc.), which isn't worth pursuing for a personal single-user tool. So treat this as expected, recurring maintenance rather than a one-time bug to fix.

**When the token expires, get a fresh one:**
```bash
rclone config reconnect gdrive: --config <path-to-your-rclone.conf>
```
If `reconnect` doesn't work, do a full `rclone config` re-auth from scratch instead (same manual/headless flow as the original setup).

Along the way you'll see a **"Google hasn't verified this app"** warning — expected every time, for the same restricted-scope reason above. Click **Advanced → Go to \<app name\> (unsafe)** to proceed.

Verify it worked: `rclone lsd gdrive: --config <path>` should list folders with no error.

If this cadence is too disruptive for your workflow, the alternative is rclone's default shared client (no custom OAuth client at all) — it doesn't hit this 7-day cap, but is heavily rate-limited instead (see the HF sync troubleshooting note in step 6 below for what that looks like in practice). There's no way to get both a long-lived custom-client token and skip full verification for this scope.

#### Making the rclone config reproducible across sessions

Since each new Colab session starts from a clean VM, redoing the `rclone config` wizard every time is painful. Which persistence method applies depends on whether you're driving Colab through the CLI (as the rest of this guide does) or through the notebook UI — they are **not** interchangeable: `google.colab.userdata` is a notebook-frontend API and is unreachable from a bare `colab console`/`colab exec` shell session, so it doesn't work for the CLI-driven workflow this guide otherwise uses.

##### CLI-native: keep the file locally, push it with `colab upload`

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

##### Notebook UI alternative: Colab Secrets

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

### 6. Downloading datasets from Hugging Face

This doesn't need a GPU and isn't really part of environment setup, so **run it on your local machine, not inside the Colab session** — better bandwidth and persistent storage than an ephemeral VM, and it doesn't tie up a billed GPU session while a large dataset transfers. Point it at a locally-mounted Drive path (via your own `rclone mount`, independent of the Colab-session `gdrive:` remote this guide otherwise uses):

```bash
uvx --from huggingface_hub hf buckets sync \
      hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002 \
      ~/google_drive/vesuvius/ink-dataset/phercparis4/w00_20231016151002
```

If you'd rather run it from inside the Colab session anyway (e.g. no local Drive mount set up), the same command works against the session's mount instead — just swap the destination for `/content/drive/ink-dataset/phercparis4/w00_20231016151002`.

#### Troubleshooting: `RuntimeError: ... IO Error: Input/output error (os error 5)` partway through

Hit this on a real run, at ~90% through a multi-GB sync into a locally-mounted `rclone` Drive. Root cause: the local rclone remote was using **rclone's default shared OAuth client** (no `client_id`/`client_secret` in its config section) — the same shared client this guide already warns is heavily rate-limited (see the "Mounting Google Drive" section above), and which we independently hit `Quota exceeded ... Requests per minute` on during ordinary testing. A bulk sync makes far more API calls than casual use, so it can trip that same per-minute quota partway through a large transfer. FUSE has no rich error channel for "your API quota was exceeded" — it just surfaces a generic `EIO` (`os error 5`) to whatever's writing, which is what Xet's file-reconstruction step hit here.

- **To finish the download**: just re-run the exact same `hf buckets sync` command. It diffs source vs. destination first (see below), so it should skip everything already downloaded and only fetch the remaining gap — much less API load, much less likely to re-trip the same limit.
- **To avoid this on future large syncs**: set up your own OAuth client for whichever local `rclone` remote you're syncing into, the same way this guide has you do for the Colab-side `gdrive:` remote — follow rclone's [own instructions](https://rclone.org/drive/#making-your-own-client-id) and `rclone config reconnect <remote>:` with the new client ID/secret. The shared client is fine for occasional small operations; it isn't for bulk dataset transfers.

#### Verifying the sync actually completed

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

### 7. Reading input data: Google Drive vs. public S3

For datasets already published to the [Vesuvius Challenge open-data S3 bucket](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html), `vesuvius`'s `ink_detection` data layer can read directly from `s3://` at training time — no download, no sync, no local copy — as an alternative to mounting Drive and syncing the dataset there first. This only applies to data actually published on that bucket; your own private data (and all training *outputs* — checkpoints, logs, run configs) still needs Drive or another writable store, since the bucket is read-only to you.

#### Data read-speed results (2026-09-22)

Measured from inside a live Colab session (T4), reading real files from each source:

| Source | File | Size | Time | Throughput |
| --- | --- | --- | --- | --- |
| S3 (anonymous, direct HTTP GET) | `PHerc0139/.../ink-detection/*.tif` | 40.8 MB | 1.35s | **~30 MB/s** |
| Google Drive (`rclone`, direct API, no FUSE mount) | segment `x.tif` | 66.7 MB | 6.07s | **~11 MB/s** |

S3 was ~2.7x faster than Drive in this comparison — the point of this comparison is Colab-to-cloud-storage speed specifically, not a general claim about S3 vs. Drive.

#### Enabling it: fully-explicit dataset entries

`DatasetSource` (in `src/vesuvius/ink_detection/config.py`) supports a fully-explicit mode that skips local directory discovery entirely: give it `segment_names` plus `surface_volume_paths`, `inklabels_paths`, and `supervision_mask_paths` — each an `{segment_name: path}` map, where `path` can be `s3://...`, `https://...`, or a local path. `segments_path` is still required by the config schema but is never touched in this mode (safe to set to any placeholder value). See [`configs/ink_tutorial_s3_test.json`](../configs/ink_tutorial_s3_test.json) for a working example against a real published segment (confirmed: 2,611 patches discovered and a 100-iteration training smoke test completed successfully reading only from S3).

Two things worth knowing if you point this at S3 data:

- **Anonymous access only works for the specific bucket substring `vesuvius-challenge-open-data`** — this is a hardcoded allowlist in `volume_io.py`'s `open_volume_root`, not a generic "any public S3 bucket" credential-free path.
- **Some published segments store `supervision`/`inklabels` as a single unchunked flat 2D array** (no z-stack, unlike the 3D z-stack convention this pipeline otherwise assumes) — the data layer now handles both automatically. If a segment's labels are one big unchunked blob, also set `volume_cache_dir`/`volume_cache_max_gb` in the config (an existing generic disk-cache layer) so the label array is fetched from S3 once and served from local disk on every subsequent patch read, instead of re-fetching the whole blob on every dataloader sample — this was the difference between a training step taking ~2.4s and the process appearing to hang indefinitely.

### Troubleshooting: console appears stuck during installs

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

### Known issues

- **`volume-cartographer` install/compile failures on older Ubuntu (e.g. Jammy 22.04, which both local dev machines and Colab may run).** `volume-cartographer/scripts/install_build_deps.sh` originally assumed packages only available on the newer Ubuntu release CI builds against (`ubuntu:26.04`): LLVM 21 (`flang-21`/`libclang-rt-21-dev`), CMake ≥3.28, and a GCC new enough for C++23's `std::byteswap` (GCC ≥13). The script now detects and works around all three automatically (adding `apt.llvm.org` and `ubuntu-toolchain-r/test` as needed, and `vesuvius/pyproject.toml`'s `cmake.version` constraint was corrected to match `CMakeLists.txt`'s real minimum) — these are no-ops wherever the OS already has new-enough versions. There is no longer a separate Colab-only extra: `models` includes `volume-cartographer` again, so step 3 above always needs `install_build_deps.sh` run first.
- **`colab drivemount` and Colab Secrets (`google.colab.userdata`) do not work from a CLI-driven session.** Confirmed against a live session: `drivemount`'s `google.colab.drive.mount()` call times out waiting on an OAuth handoff, and `userdata.get()` raises `google.colab.userdata.TimeoutException: ... Secrets can only be fetched when running from the Colab UI.` Both require a live Colab browser tab connected to the runtime, which a bare `colab console`/`colab exec` session never has — this is independent of the `jupyter-kernel-client` bug noted in step 1. Use the CLI-native `rclone` + `colab upload`/`colab download` approach in the Drive-mounting section instead.

If a command genuinely looks stuck rather than just slow, check whether it's still making progress (e.g. re-run with `uv sync -v`, or check network activity) before assuming it's hung — some of these downloads are large and Colab's outbound bandwidth is not always fast.
