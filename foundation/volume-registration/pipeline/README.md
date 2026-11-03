# Registration pipeline

The production version of the registration review process; see `DESIGN.md` for what it is
and why. Built so far:

- **step 1**: the record and the migration of the first runs;
- **step 2, in part**: the cores, merged in unchanged except for imports, with their tests;
- **step 3**: the engine, the stage table and the command;
- **step 4**: every automatic stage. Capture check, hold-out draw and measurement, sampling
  and fitting run on `cores/fit_session.py`; scoring and storing run on the record.

Still to do: **step 5**, the review UI (judging the hold-out set and the blind round; until
then answers go in with `volreg answer`), and **step 6**, the container. The capture check's
thresholds (12 probes, at least half matching 0.85) are a first guess from the first runs.

Two things behave differently from the first runs:

- **A hold-out measurement is frozen once drawn.** The first runs re-measured test patches on
  every fit pass, so a measurement could move after it had been judged. The fit stage now
  measures fit patches only (`correlate(skip_test=True)`, the one change to the cores).
- **The block sampler adds every textured block in a read batch.** So it can draw more than
  asked: 21 probes for 12 on PHerc0814. That is unchanged from the first runs, and harmless.

## What is here

| | |
|---|---|
| `volreg/record.py` | the per-pair record: its files, whole-file writes, the append-only event log, the reviewer |
| `volreg/scoring.py` | score a transform against the hold-out set, in fine voxels, from the record alone; the publishing bar |
| `volreg/migrate.py` | convert a session of the first runs (`fit_session.py`) into a record |
| `volreg/engine.py` | state read off the files, as make does: done, stale, running, failed, ready, waiting, blocked; background jobs; redo |
| `volreg/stages.py` | the stage table of `DESIGN.md`, and what each stage does |
| `volreg/cli.py` | `run`, `status`, `answer`, `more`, `redo`, `job`, `check`, `migrate` |
| `volreg/cores/` | algorithms merged in from the research modules, unchanged except for imports: `deformable.py`, `transform_fit.py`, `dense_correlation.py`, the transform file I/O, point fits and volume reading of `transform_utils.py`, and `read_patch` of `blob_snap.py`, `fit_session.py` and `session_overview.py` |

## Run

From this directory. The reviewer comes from `--handle` and `--email`, or from
`VOLREG_HANDLE` and `VOLREG_EMAIL`:

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli run PAIR
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli answer PAIR p0007 lines-up
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli more PAIR --holdout 20
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli redo PAIR fit

    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli status RECORDS
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli check RECORDS
    uv run --python 3.12 --with-requirements requirements.txt python -m volreg.cli migrate \
        SESSION OUT --handle HANDLE --email EMAIL --staged PUBLISH_CANDIDATES

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
