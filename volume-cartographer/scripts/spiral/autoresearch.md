# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD (any working copy changes should be left intact, assuming they are not in `fit_spiral.py`).
3. **Read the in-scope files**: Only this 'atlas' subfolder of the repo is relevant. Read these files for full context:
   - `tifxyz.py` — helper for loading grid-topo quad-mesh patches.
   - `fit_spiral.py` — the file you modify. Losses, optimization, flow model, etc.
4. **Initialize results.tsv**: Create `results_mar5.tsv` with just the header row. The baseline will be recorded after the first run. Change `mar5` to whatever branch tag is chosen.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The fitting script runs for a **fixed iteration budget of 10000 steps**; this will take a few minutes in the baseline configuration (and you should avoid increasing this time by too large a factor).
You launch it as: `WANDB_MODE=disabled python fit_spiral.py`.

**What you CAN do:**
- Modify `fit_spiral.py` — this is the only file you edit. Everything is fair game: losses, optimizer, hyperparameters, flow field, etc -- but NOT the metrics themselves! That would be cheating.

**What you CANNOT do:**
- Change the number of iterations (num_training_steps in config) above 10000. It must stay fixed (or can be reduced if that actually gives better results).
- Install new packages or add dependencies. You can only use what's already installed in this conda env.
- Modify the evaluation metric (get_patch_satisfied_areas and related methods) or its parameters (in metrics_config), or the working-set growth logic (working_set_check_interval and how the working set expands).

**The goal is simple: get the largest working set by the end of training.** The working set grows progressively as patches become satisfied; the script logs each addition to `working-set.txt` in the run's output directory, ending with a `training ended: final working set size N/M` line. That final `N` is the primary metric. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the losses. The only constraints are that the code runs without crashing, does not run for more than 10000 iterations, and wall-clock time does not increase to more than three times the baseline run.

**Hint — early growth matters**: Look at *when* patches enter the working set in `working-set.txt`, not just the final count. Seeing many patches added EARLY in training (low step numbers, growing densely in the first ~quarter of the run) is a strong positive signal — it means the optimizer is finding good geometry quickly and the run has lots of remaining budget to absorb harder patches. Conversely, a run where additions are sparse early and only pick up near the end is fragile: it likely got lucky and probably won't generalize across seeds. When two runs reach a similar final `N`, prefer the one whose early-step density is higher. Consider aborting runs early if their working set is clearly growing much slower than prior runs.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny working-set improvement that adds 20 lines of hacky code? Probably not worth it. A tiny working-set improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is. After that, it might be a good idea to do simple hyperparameter changes before anything more drastic.

**Stochasticity**: The code is VERY sensitive to the random seed and CUDA non-determinism. We prefer changes that are robust to the choice of seed (and across runs with the same seed), not those which only work for one specific seed. So seemingly beneficial changes should be verified across different seeds/runs.

## Output format

The script writes a `working-set.txt` file into the run's output directory (something like `out/<date>_<scroll>_..._<runtag>/working-set.txt`). It contains one line per working-set change and a final summary line. Example:

```
step 0: initialised with seed patch <id> (size 1/345)
step 120: added patch <id> (size 2/345)
step 350: added patch <id> (size 3/345)
...
training ended: final working set size 47/345
```

The last line is what matters — extract the final working set size with:
```
grep "training ended" out/*/working-set.txt | tail -n 1
```

(or use the specific run's directory if multiple runs are present). If the file has no `training ended` line, the run did not finish cleanly. Note depending on the computing platform the numbers might look different.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	final_working_set	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. final working set size (e.g. 47) — use 0 for crashes. This is the metric.
3. wall-clock time in seconds
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	final_working_set	time_s	status	description
a1b2c3d	12	300	keep	baseline
b2c3d4e	18	320	keep	increase LR to 0.04
c3d4e5f	22	254	keep	switch to GeLU activation
d4e5f6g	0	289	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `fit_spiral.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `WANDB_MODE=disabled python fit_spiral.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the result: find the run's output directory and grep `working-set.txt` for the `training ended` line to get the final working set size.
6. If there is no `training ended` line, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Compare: if the final working set size improved (higher), advance the branch keeping the commit. Otherwise, git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: If a run exceeds three times the duration of the baseline, kill it and treat it as a failure (discard and revert).

**Shell commands**: Only inspect `run.log` to check current run status, and poll internally for whether the process is still running. Do not use external tools like `pgrep` that might require permission from the user. DO NOT USE ANY SHELL COMMAND REQUIRING SUBSTITUTION (like `ps $(pgrep ...)`), since these require user confirmation (so the experiment would stop). Similarly do not use ansi c-strings in shell commands.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
