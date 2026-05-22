# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD (any working copy changes should be left intact, assuming they are not in `fit_spiral.py`).
3. **Read the in-scope files**: Only this 'spiral' subfolder of the repo is relevant. Read these files for full context:
   - `fit_spiral.py` — the file you modify. Losses, optimization, flow model, etc.
   - `tifxyz.py` — helper for loading grid-topo quad-mesh patches.
   - `point_collection.py` — helper for loading sets of annotated points, and linking them to nearby patches.
4. **Initialize results.tsv**: Create `results_mar5.tsv` with just the header row. The baseline will be recorded after the first run. Change `mar5` to whatever branch tag is chosen.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The fitting script runs for a **fixed iteration budget of 10000 steps**; this will take around 20 minutes in the baseline configuration (and you should avoid increasing this time by too large a factor).
You launch it as: `WANDB_MODE=disabled python fit_spiral.py > run.log`.

**What you CAN do:**
- Modify `fit_spiral.py` — this is the only file you edit. Everything is fair game: losses, optimizer, hyperparameters, flow field, etc -- but NOT the metrics themselves! That would be cheating.

**What you CANNOT do:**
- Change the number of iterations (num_training_steps in config) above 10000. It must stay fixed (or can be reduced if that actually gives better results).
- Install new packages or add dependencies. You can only use what's already installed in this conda env.
- Modify the evaluation metrics (get_patch_satisfied_areas, get_unattached_pcl_satisfied_counts, and related methods) or their parameters (in metrics_config).
- Change from the 'global' `working_set_mode` to another. Only the global mode is in scope here.

**The goal: improve a balance of `satisfied_patches` and `satisfied_unattached_pcls`.** These are the two primary metrics. These can be extracted from the process output using grep, if you redirect it to a file.

We also track two "softer" versions of these metrics: `satisfied_area` (a continuous-area version of `satisfied_patches`) and `satisfied_unattached_pcl_points` (a per-point version of `satisfied_unattached_pcls`). These are often more sensitive — improvements may show up here first before they crystallise into the harder metrics — so they are useful as a leading signal.

We expect all four metrics to be reasonably well correlated, but **avoid changes that make any of them substantially worse**, even if one of the others improves. A change that lifts `satisfied_patches` by 1 while halving `satisfied_unattached_pcls` is not a win.

Everything else is fair game: change the diffeomorphism, the optimizer, the hyperparameters, the losses. The only constraints are that the code runs without crashing, does not run for more than 10000 iterations, and wall-clock time does not increase more than 50% versus the baseline run.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny working-set improvement that adds 20 lines of hacky code? Probably not worth it. A tiny working-set improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is. After that, it might be a good idea to do simple hyperparameter changes before anything more drastic.

**Stochasticity**: The code is VERY sensitive to the random seed and CUDA non-determinism. We prefer changes that are robust to the choice of seed (and across runs with the same seed), not those which only work for one specific seed. So seemingly beneficial changes should be verified across different seeds/runs.

## Output format

The four metrics we care about are printed to stdout near the end of the run, in the `run.log`:

```
satisfied_patches = 47/345 (13.6%)
satisfied_area = 12345.6/98765.4 (12.5%)
satisfied_unattached_pcls = 8/20 (40.0%)
satisfied_unattached_pcl_points = 412/1023 (40.3%)
```

Extract these from `run.log` after the run completes, e.g.:
```
grep -E "^(satisfied_patches|satisfied_area|satisfied_unattached_pcls|satisfied_unattached_pcl_points) =" run.log | tail -n 4
```

If the four lines aren't present, the run did not finish cleanly. (Note: depending on the computing platform the absolute numbers might look different — only relative comparisons against the baseline on the same machine matter.)

`working-set.txt` is still useful as a diagnostic for *when* in training patches become satisfied (see the "early growth matters" hint above), even though the final size itself isn't the metric.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```
commit	satisfied_patches	satisfied_area	satisfied_unattached_pcls	satisfied_unattached_pcl_points	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. `satisfied_patches` count (e.g. 47) — use 0 for crashes
3. `satisfied_area` (float, e.g. 12345.6) — use 0 for crashes
4. `satisfied_unattached_pcls` count (e.g. 8) — use 0 for crashes
5. `satisfied_unattached_pcl_points` count (e.g. 412) — use 0 for crashes
6. wall-clock time in seconds
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	satisfied_patches	satisfied_area	satisfied_unattached_pcls	satisfied_unattached_pcl_points	time_s	status	description
a1b2c3d	12	3210.4	5	287	300	keep	baseline
b2c3d4e	18	4580.1	6	305	320	keep	increase LR to 0.04
c3d4e5f	22	5102.7	5	291	254	keep	switch to GeLU activation (pcls flat — borderline)
d4e5f6g	0	0	0	0	289	crash	double model width (OOM)
```

When deciding `keep` vs `discard`, judge across all four metrics together: any metric falling substantially below baseline is grounds for `discard`, even if another metric improved a bit. The two hard counts (`satisfied_patches`, `satisfied_unattached_pcls`) are the primary signals; the two softer values (`satisfied_area`, `satisfied_unattached_pcl_points`) are the leading signals.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `fit_spiral.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `WANDB_MODE=disabled python fit_spiral.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the result: grep `run.log` for the four `satisfied_*` summary lines (see Output format above).
6. If the four summary lines are not present, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Compare against the baseline / previous best: advance the branch keeping the commit only if the change looks like a net improvement across the four metrics (and none of them are substantially worse). Otherwise, git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: If a run exceeds three times the duration of the baseline, kill it and treat it as a failure (discard and revert).

**Shell commands**: Only inspect `run.log` to check current run status, and poll internally for whether the process is still running. Do not use external tools like `pgrep` that might require permission from the user. DO NOT USE ANY SHELL COMMAND REQUIRING SUBSTITUTION (like `ps $(pgrep ...)`), since these require user confirmation (so the experiment would stop). Similarly do not use ansi c-strings in shell commands.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
