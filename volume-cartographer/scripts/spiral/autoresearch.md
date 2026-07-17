# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jul9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD (any working copy changes should be left intact, assuming they are not in `fit_spiral.py` or its helper modules).
3. **Read the in-scope files**: Only this 'spiral' subfolder of the repo is relevant. Read these for full context before you start:
   - `fit_spiral.py` — the main file you modify. Losses, optimization, flow model, config, etc.
   - `point_collection.py` — helper for loading sets of annotated points, and linking them to nearby patches.
   - `losses.py`, `spiral_helpers.py`, `geom_utils.py`, `transforms.py`, `flow_fields.py` — fitting-side helper modules on the fit path (`flow_fields.py` is used via `transforms.py`) and NOT used by the metric/render pipeline. These are also fair game to edit (see scope below).
   - `tifxyz.py` — helper for loading/saving grid-topo quad-mesh patches. Read it for context, but it is **frozen** (see scope): it is shared with `render_ink.py`, so editing it changes the metric.
   - `run_single.py` — the pipeline runner you launch (fit → render ink → score ink). You do NOT edit this or the scoring scripts; just understand what it does.
4. **Initialize results.tsv**: Create `results_jul9.tsv` with just the header row (see "Logging results"). The baseline will be recorded after the first run. Change `jul9` to whatever branch tag is chosen.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The goal

**Maximise the area of ink recovered.** After each fit, the fitted meshes are rendered against an ink-prediction volume (`render_ink.py`) and a 2D nnU-Net scores how much ink-like surface was recovered (`get_ink_coverage.py`). The single number we optimise is:

> **`total_fg_pixels`** — the total count of ink-foreground pixels across all rendered winding strips. Higher = more ink recovered. This is the metric.

A better fit — one that produces a more coherent, correctly-flattened surface that lands on the inked papyrus — recovers more ink. That is the whole game: change the *fitting* so the resulting surface exposes more ink.

We also track **`overall_fg_fraction`** (foreground pixels ÷ total strip pixels) as a **secondary / sanity metric**. It guards against gaming: if a change balloons `total_fg_pixels` only by inflating the surface with garbage geometry, the fraction will collapse. Treat a big `total_fg_pixels` gain that comes with a large fraction *drop* with suspicion — that is probably noise, not ink. A change that lifts both, or lifts total while holding fraction roughly steady, is a real win.

**Also keep an eye on the satisfaction metrics.** `fit_spiral.py` prints some geometry-fit metrics — `satisfied_patches`, `satisfied_area`, `satisfied_unattached_pcls`, `satisfied_unattached_pcl_points` — near the end of its log (`<out_dir>/logs/<tag>.fit.log`). These are **not the objective** (ink coverage is), but they are a useful *diagnostic* for what a change did to the underlying fit. Track how they move alongside ink coverage: if ink coverage climbs while the satisfaction metrics hold up (or improve), you have a genuinely better fit; if ink coverage climbs while the satisfaction metrics fall off a cliff, be suspicious that you are contorting the surface to catch stray ink rather than fitting the scroll better. They are a cross-check, not a target — do not optimise them directly, and never let them override the ink-coverage decision.

## Scope — what you can and cannot change

**What you CAN do:**
- Modify `fit_spiral.py` **and the fitting-side helper modules on the fit path that the metric/render pipeline does not use** — namely `losses.py`, `spiral_helpers.py`, `geom_utils.py`, `transforms.py`, `flow_fields.py`, `point_collection.py` (and, if you need them, the other fit-only imports: `sample_spiral.py`, `tracks.py`, `umbilicus.py`, `ddp_helpers.py`, `lasagna_data.py`). Everything about the *fit* is fair game: losses, optimizer, hyperparameters, flow field, diffeomorphism, initialization, sampling, etc.
- Create/modify your own helper *shell/python scripts* for driving the loop (batch launchers, summarizers, etc.).

**What you CANNOT do:**
- Touch the **metric / render pipeline**. `render_ink.py`, `get_ink_coverage.py`, the nnU-Net ink model / checkpoint, the ink volume, and the render/score parameters (scale, num-slices, fg-threshold, flatten settings) are all **frozen**. Modifying any of them — or otherwise engineering the score rather than the fit — is cheating.
- Edit **`tifxyz.py`**. It is shared: `render_ink.py` imports it to load the meshes it renders, so any change to it changes the metric. It is frozen even though `fit_spiral.py` also uses it. (`satisfaction_metrics.py` is likewise best left alone — editing it cannot affect the ink score.)
- Change the number of fit iterations (`num_training_steps` in the config, currently `30000`) **above** its current value. It must stay fixed (it *can* be reduced if that genuinely gives equal-or-better ink coverage).
- Install new packages or add dependencies. Use only what is already in this conda env.

**Wall-clock constraint**: The fit portion should not get more than ~50% slower than the baseline fit. (The render+score portion is roughly fixed cost and outside your control; don't worry about it beyond not making the mesh pathologically large.) If a full run exceeds three times the baseline run's duration, kill it and treat it as a failure.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal-or-better ink coverage is a great outcome — a simplification win. A tiny gain that adds 20 lines of hacky code? Probably skip. A gain (or a wash) from *deleting* code? Keep.

**The first run**: Your very first run should always establish the baseline — run the pipeline as-is, unmodified. After that, prefer simple hyperparameter changes before anything drastic.

**Stochasticity**: The code is sensitive to the random seed and CUDA non-determinism. Prefer changes that are robust across seeds/runs, not ones that only help for one specific seed. Since you run two at a time, a cheap robustness check is to run the same change under two seeds concurrently and see if the ink gain survives. Seemingly-beneficial changes should be verified across seeds/runs before you commit to them.

## The pipeline and how to run it

Each experiment is one full pipeline run, driven by `run_single.py`, which chains three steps:

1. `torchrun --nproc_per_node=<n> fit_spiral.py` — fits the meshes.
2. `render_ink.py <meshes_dir>` — renders ink strips into `<meshes_dir>/ink`.
3. `get_ink_coverage.py <meshes_dir>/ink` — scores ink coverage into `<meshes_dir>/ink_metric`.

`run_single.py` reads three things from the environment, which the caller sets:

- `CUDA_VISIBLE_DEVICES` — the GPU subset for this run. `run_single.py` honours it: `--nproc_per_node` is set to the number of visible GPUs, and the pin is passed straight through to every step.
- `FIT_SPIRAL_RUN_TAG` — the run tag (names the output dir and the fitted-mesh folder).
- `FIT_SPIRAL_OUT_DIR` — the base output dir.

**Run two experiments at a time, each on four GPUs.** On an 8-GPU box that means one run pinned to `CUDA_VISIBLE_DEVICES=0,1,2,3` and one to `4,5,6,7`, launched concurrently. Each run is fully self-contained on its four GPUs (fit, render, and score all stay within the pin), so the two never collide.

Per-step logs go to `<out_dir>/logs/<tag>.{fit,ink,coverage}.log`, and the ink metric is written to `<out_dir>/<datedir>_<tag>/meshes/fitted_<tag>/ink_metric/metrics.json`.

## Output format — reading the metric

`get_ink_coverage.py` prints a summary near the end of its log (`<out_dir>/logs/<tag>.coverage.log`):

```
================================================================
INK SURFACE-AREA METRIC
================================================================
strips scored          : 42
TOTAL ink foreground   : 1,234,567 px   (metric; higher = more ink)
total strip area       : 9,876,543 px
overall ink fraction   : 12.500 %
```

and writes the same numbers as JSON to `.../ink_metric/metrics.json`:

```json
{
  "summary": {
    "num_strips": 42,
    "total_fg_pixels": 1234567,
    "total_pixels": 9876543,
    "overall_fg_fraction": 0.125
  },
  "strips": [ ... ]
}
```

**Read the metric from `metrics.json`** (robust to log formatting) — the primary field is `summary.total_fg_pixels`, the guard is `summary.overall_fg_fraction`. If `metrics.json` is missing, the run did not finish cleanly; check the three per-step logs (`.fit.log`, `.ink.log`, `.coverage.log`) for the failure. (Absolute numbers depend on the machine; only relative comparisons against the baseline on the same machine matter.)

## Logging results

When an experiment is done, log it to `results_<tag>.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Do NOT commit this file; leave it untracked.

Header + 9 columns:

```
commit	total_fg_px	ink_fraction	num_strips	sat_patches	sat_pcls	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. `total_fg_pixels` (e.g. 1234567) — use 0 for crashes
3. `overall_fg_fraction` as a percentage (e.g. 12.5) — use 0 for crashes
4. `num_strips` scored (e.g. 42) — sanity check that the render produced the expected strips; use 0 for crashes
5. `satisfied_patches` % from the fit log (e.g. 13.6) — diagnostic cross-check, not a target; use 0 for crashes
6. `satisfied_unattached_pcls` % from the fit log (e.g. 40.0) — diagnostic cross-check, not a target; use 0 for crashes
7. wall-clock time in seconds (whole pipeline)
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Columns 5–6 just record how the geometry fit moved (grep them from `<out_dir>/logs/<tag>.fit.log`) so you can eyeball them next to ink coverage; they never decide `keep`/`discard` on their own.

Example:

```
commit	total_fg_px	ink_fraction	num_strips	sat_patches	sat_pcls	time_s	status	description
a1b2c3d	1180000	11.8	42	13.6	40.0	1850	keep	baseline
b2c3d4e	1265000	12.6	42	14.1	41.2	1900	keep	increase LR to 0.04 (ink +7%, sat steady)
c3d4e5f	1310000	9.1	58	6.2	22.0	1980	discard	looser winding tol — more strips but fraction+sat tanked (noise)
d4e5f6g	0	0	0	0	0	410	crash	double model width (OOM)
```

When deciding `keep` vs `discard`: `total_fg_pixels` is the primary signal; `overall_fg_fraction` is the guard; the two satisfaction columns are a diagnostic cross-check. A `total_fg_pixels` gain with a stable-or-rising fraction is a keep. A `total_fg_pixels` gain bought with a large fraction drop — or with the satisfaction metrics collapsing — is likely gaming/noise → discard. Prefer changes whose gain reproduces across seeds.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jul9`).

LOOP FOREVER (two experiments in flight at a time):

1. Look at the git state: the current branch/commit we're on.
2. Tune the fitting code (`fit_spiral.py` and/or its helper modules) with an experimental idea by directly hacking the code, or by setting environment variables. You can put two variants (or one variant under two seeds) into the two concurrent slots.
3. git commit the change(s).
4. Launch the batch: two `run_single.py` runs, one on GPUs `0,1,2,3` and one on `4,5,6,7`, each with its own `FIT_SPIRAL_RUN_TAG`. Redirect all output to files — do NOT tee or let output flood your context.
5. Read out each result: load `summary.total_fg_pixels` and `summary.overall_fg_fraction` from each run's `metrics.json` (see Output format).
6. If a run's `metrics.json` is absent, it crashed. Read the tail of its per-step logs to find the stack trace and attempt a fix. If you can't get it working after a few attempts, give up on that idea.
7. Record each result in the tsv (do NOT commit the tsv; leave it untracked).
8. Compare against the baseline / previous best: advance the branch keeping a commit only if it's a net improvement (`total_fg_pixels` up, fraction not substantially worse, gain plausibly robust). Otherwise `git reset` back to where you started.

You are a completely autonomous researcher. If an idea works, keep; if not, discard; and you advance the branch so you can iterate. If you feel stuck you can rewind, but do this very sparingly (if ever).

**Shell commands — avoid approval prompts at all costs**: Every shell command you type is matched against an allow-list, and anything novel triggers a user-approval prompt that *blocks the whole loop* until the (possibly absent/asleep) user responds. So the iron rule is: the set of *distinct command strings* you ever type must be tiny and fixed. Concretely:

- DO NOT type any command containing shell substitution (`$(...)`, backticks), process substitution (`<(...)`), or ANSI C-strings (`$'...'`). These always prompt. This includes inside `run_in_background` commands and inside `&&`/`;` chains. (Operators like `&&`, `|`, `>` themselves are fine; it is *substitution* that prompts.)
- DO NOT type ad-hoc inspection one-liners (`for d in ...; do grep ... $(...); done`, `cat .../metrics.json | ...`, `python -c "import json; ..."`, etc.). Every new variant is a new prompt, and JSON-parsing one-liners are especially tempting here — resist them.
- INSTEAD, keep everything down to **two fixed-form helper scripts plus one fixed git command**, and invoke each with an *identical command string* every time. All the logic that needs substitution/loops/pipes/JSON-parsing lives *inside* the script file (substitution inside a committed `.sh`/`.py` is fine — only the command string you submit is matched):
  - `launch.sh` — reads a `batch.txt` manifest (`<gpus> <tag>` per line, e.g. `0,1,2,3 jul9a` / `4,5,6,7 jul9b`); for each line it exports `CUDA_VISIBLE_DEVICES`, `FIT_SPIRAL_RUN_TAG`, `FIT_SPIRAL_OUT_DIR` (and `WANDB_MODE=disabled`) and runs `python run_single.py` in the background with all output redirected to the per-step logs, then `wait`s for both. One `bash launch.sh` launches the whole batch; one completion notification covers it.
  - `summarize.sh` — scan each run's `metrics.json` and fit logs and print a TSV row (total_fg_px, fraction, num_strips, sat_patches, sat_pcls). Use this instead of ad-hoc `grep`/`cat`/`python -c` over the JSON.
  - Commit with the fixed string `git add -A && git commit -F COMMIT_MSG.txt` — no wrapper script needed, because the *command* is already identical every time (only `COMMIT_MSG.txt`'s contents change). A raw `git commit -m "..."` with a unique message would be a new string and prompt, so always route the message through the file.
- CREATE AND EDIT ALL DATA FILES WITH THE Write/Edit TOOLS, never with `printf >`/`echo >`/`mkdir` in Bash. This includes **appending each result row to `results_<tag>.tsv`** — do it with Edit, not a shell script. The Write/Edit tools never trigger a Bash approval prompt and avoid minting a unique command string per file. Same for `batch.txt`, `COMMIT_MSG.txt`, any config-override files, and the two helper scripts themselves.
- To inspect progress, prefer reading the background task's output / a run's log or `metrics.json` with the Read tool, or run `summarize.sh`. Do not use external tools like `pgrep`/`ps`/`nvidia-smi` in ad-hoc form (also prompt). Poll internally / rely on the background-completion notifications.

The very first time you use the experiment loop, write these two scripts (with the Write tool) before launching anything, so that from then on every shell command is one of a handful of pre-approved fixed strings and the loop can run unattended without ever stopping for approval.

**Crashes**: If a run crashes (OOM, a bug, etc.), use judgment: if it's something dumb and easy (a typo, missing import), fix and re-run. If the idea is fundamentally broken, skip it, log `crash`, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expecting you to work *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — read papers, re-read the in-scope files for new angles, combine previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Two runs at a time, each taking ~30 minutes, is ~4 experiments/hour — roughly 30 over a night. The user wakes up to a night's worth of experimental results, all completed while they slept.
