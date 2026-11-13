# Registration pipeline

The production version of the registration review process; see `DESIGN.md` for what it is
and why. Built so far:

- **step 1**: the record and the migration of the first runs;
- **step 2, in part**: the cores, merged in unchanged except for imports, with their tests --
  and one change since: `deformable.fold_errors` reads each held-out patch's error off the
  full fit instead of refitting without it (the same numbers to 1e-12, one solve per
  smoothing level instead of one per patch);
- **step 3**: the engine, the stage table and the command;
- **step 4**: every automatic stage. Capture check, hold-out draw and measurement, sampling
  and fitting run on `cores/fit_session.py`; scoring and storing run on the record.

- **step 5**: the review UI, one server for every pair (see below).

Still to do: **step 6**, the container. The capture check's thresholds (12 probes, at least
half matching 0.85) are a first guess from the first runs.

Two things behave differently from the first runs:

- **A hold-out measurement is frozen once drawn.** The first runs re-measured test patches on
  every fit pass, so a measurement could move after it had been judged. The fit stage now
  measures fit patches only (`correlate(skip_test=True)`, the one change to the cores).
  Since a patch measured under a rough start could move once the fit is better, the
  `recheck` stage measures the hold-out set again under each fit, changes nothing, and
  records how far each would move. Lining-up patches that would move more than half a fine
  voxel are listed to be judged again. On PHerc0814 (2.4 to 9.36 µm) they move 0.10 fine
  voxels median and 0.23 at most.
- **The hold-out stage renders what a person judges, once.** The images are stored under
  `work/images/`, and their digests go into `holdout.json`. The review page only reads them,
  so judging needs no volume access.
- **The block sampler adds every textured block in a read batch.** So it can draw more than
  asked: 21 probes for 12 on PHerc0814. That is unchanged from the first runs, and harmless.

## What is here

| | |
|---|---|
| `volreg/record.py` | the per-pair record: its files, whole-file writes, the append-only event log, the reviewer |
| `volreg/scoring.py` | score a transform against the hold-out set, in fine voxels, from the record alone; the publishing bar |
| `volreg/migrate.py` | convert a session of the first runs (`fit_session.py`) into a record |
| `volreg/release.py` | gather the newest decided record of every pair into one release set |
| `volreg/engine.py` | state read off the files, as make does: done, stale, running, failed, ready, waiting, blocked; background jobs; redo |
| `volreg/stages.py` | the stage table of `DESIGN.md`, and what each stage does |
| `volreg/cli.py` | `run`, `status`, `answer`, `more`, `redo`, `adopt`, `job`, `check`, `migrate`, `import` |
| `volreg/cores/` | algorithms merged in from the research modules, unchanged except for imports and the held-out errors of `deformable.fold_errors`: `deformable.py`, `transform_fit.py`, `dense_correlation.py`, the transform file I/O, point fits and volume reading of `transform_utils.py`, and `read_patch` of `blob_snap.py`, `fit_session.py` and `session_overview.py` |

## Stages

| stage | who | |
|---|---|---|
| `capture` | auto | draws probe patches under the start and fails, saying how well they matched, if too few match |
| `holdout` | auto | draws the hold-out set (no more than asked for), measures it under the start, renders it |
| `judge-holdout` | you | done when enough patches line up, or all are answered |
| `fit` | auto | samples fit patches, then correlates and fits; starts after the capture check, beside the hold-out set, so more hold-out patches do not fit again |
| `recheck` | auto | measures the hold-out set again under the fit (needs both), and flags drift |
| `score` | auto | the bend and the affine against your answers, and the bar |
| `blind` | auto | draws and renders a blind round of bend against affine on the patches lining up or partly |
| `final-review` | you | the blind picks, then the decision; publishing what misses the bar needs a reason, and is stored as an override |
| `store` | auto | writes the decided transform and its scores into the record |

A job that finishes starts whatever its result made ready, and so does an answer or a
decision on the review page. So a pair moves on by itself until you are next.

## Review UI

    VOLREG_HANDLE=... VOLREG_EMAIL=... uv run --python 3.12 --with-requirements requirements.txt \
        python -m volreg.review.server PAIRS_DIR [--records RECORDS_DIR ...] --port 8790

The page at `/` shows each pair under `PAIRS_DIR` in three parts. At the top is a compact
overview of its steps. Below it, **Continue** starts every step that can run, beside a link
to whatever waits for you. Then every step is listed with its state, its settings (the
hold-out set's size, the fit's patches and passes, the blind round's size) and a Redo
button. A changed setting is logged under your id; it and Redo start the step again right away. The judging page uses the first runs' keys: `1`–`5` to
answer, `space` to flip, `m` for mode, `n` for a note, `u` to undo, `g` to open both whole
volumes overlaid in Neuroglancer where the patch sits (`volreg/review/viewer.py`; the fine
volume placed as the picture was, the current fit's affine part as a second layer, no bend),
and `?unit=octants` to judge octants. The blind page uses `1`/`2`, `Enter` to pick, `x` for can't tell, `n` for
neither, and `j`/`k` to move. It ends in the decision. The server never reads a volume.

Pairs can be collapsed, and finished ones are hidden unless you untick the filter. `/fits`
lists every pair's fit in one table: the pipeline's pairs and, read-only, the records under
`--records` (the first runs' migrated records). Each row shows the fit's model, points and
bend; the pair's settings; where it stands; the bend's and the affine's median and p90
against the hold-out set, with the bar; the blind tally; and the decision. Click a pipeline
pair to see every fit its session made.

## Run

From this directory. The reviewer comes from `--handle` and `--email`, or from
`VOLREG_HANDLE` and `VOLREG_EMAIL`:

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli new PAIR \
        --start START_TRANSFORM --manifest MANIFEST
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli run PAIR
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli answer PAIR p0007 lines-up
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli more PAIR --holdout 20
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli redo PAIR fit

`new` starts a pair from a transform file (moving → fixed, the first runs' `dense/` or
`discovered/` ones), its volumes from the manifest entry of the same name, and runs it. Only
the start's affine part is kept.

A first run's session goes into a new pair after its capture check, before the hold-out
set is drawn: its patches with their measurements and blocks, its fits (as `prior-<id>`,
the last current), and the answers, notes and verdicts of its migrated record. The old
blind rounds and decision stay behind, since they judged a fit the pipeline has not made:

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli import \
        PAIR SESSION MIGRATED_RECORD

When a change to the code changes only what a stage needs, not what it does, `adopt` keeps
its finished result instead of running it again. The old stamp is kept, and the change is
logged with the reason:

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli adopt PAIR STAGE "why"

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli status RECORDS
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli check RECORDS
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli migrate \
        SESSION OUT --handle HANDLE --email EMAIL --staged PUBLISH_CANDIDATES

`release` gathers the newest decided record of every pair into a new folder, from any
number of records folders (the pipeline's and the migrated first runs'). A pipeline record
wins over a migrated one of the same pair. Each pair keeps `pair.json`, `holdout.json`
(every answer with its reviewer's handle and email), `final-review.json`,
`patch-reviews.json` where there is one, and `fit/`; the event log, the settings and `work/`
stay behind. `index.json` lists each pair's source, decision, score, answer counts and file
hashes, and what its record does not hold (a migrated record has no code commit or decision
time, for example); nothing is filled in. Undecided and superseded records are listed as
left out. `check` runs on a release as on any records folder:

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli release OUT RECORDS [RECORDS ...]

Tests:

    uv run --python 3.12 --with-requirements requirements.txt --with pytest python -m pytest tests

## Migrated so far (2026-11-03)

The 16 sessions of the first runs, migrated into 16 records (6.7 MB):

- 703 hold-out answers, 687 of them per patch and 16 per octant;
- 244 patch verdicts, 307 blind picks and 43 notes.

All of these are logged in `events.jsonl` too. The event log is the only source of the
answers: when `holdout.json` is rewritten from the log, every answer in all 16 records comes
out the same. `check` recomputes the published score of each
of the 10 staged transforms from its record alone (`holdout.json` and `fit/transform.json`). All
10 match the stored scores and the staged provenance exactly. One of them misses the bar and
is published on the reviewer's override, which its record states: PHerc0009B, affine only,
with a median of 2.28 fine voxels against a limit of 2.

Not migrated: blocks, cached volume reads, and fits nobody chose. All of them can be
recomputed. Blind picks in the first runs carry no time of their own, so they take their
round's time.
