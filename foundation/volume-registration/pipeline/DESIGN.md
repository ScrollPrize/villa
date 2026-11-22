# Registration pipeline — design

The production version of the registration review process worked out on branch `worktree-volreg-auto-refine` (`foundation/volume-registration/review-tool.md` there):
a person judges a hold-out set and the final fit, and everything between is automatic. This
directory is self-contained; the research modules on that branch stay the research record, and the
algorithms they hold that the process actually used are merged in here in a production pass.

Decided with the reviewer on 2026-11-03.

## What the first runs taught

- **A person's time is the scarce input**: about 60 test answers and 20 blind picks a pair.
  Everything else can be recomputed, so the human record is stored apart from derived state,
  append-only, and never rewritten in place.
- **A test answer judges a measurement, not a fit.** It stays valid across refits, new
  patches and new code; nothing in the record depends on a particular fit.
- **A bad start wastes review time.** PHerc1667's atlas start was 166 µm off; PHerc0500P2
  0.55 µm never captured. A capture check runs before anyone is asked to judge.
- **Provenance must name code and data versions**, not only the session and fit.
- **No process state in ad-hoc scripts.** The drivers that ran the first pairs lived in a
  tmpfs scratchpad and were lost (rebuilt from the transcript). Every step is a command of
  the tool.
- **Reads are chunk-aligned and the cache is disposable** (the first sessions hold 33 GB of
  cached blocks, all re-derivable). A session's `blocks/` and `overview/` carry a
  `CACHEDIR.TAG`, so a backup run with `--exclude-caches` leaves them out.
- **Scan artifacts belong in the data**: DLS slab seams, for example, are a patch flag.

## Stages

| # | stage | who | produces |
|---|---|---|---|
| 0 | pair and start transform | given | `pair.json` (an automatic start fit is out of scope for now) |
| 0b | capture check | auto | does the start land test patches; stop here if not |
| 1 | draw the hold-out set, measure it, render its review images | auto | hold-out patches, measurements, images |
| **2** | **judge the hold-out set** | **reviewer** | answers, per patch or per octant |
| 3 | sample fit patches, measure, fit | auto | fits (affine and bend); **starts right after 1**, in parallel with 2 |
| 3b | measure the hold-out set again under the fit | auto | how far each frozen measurement would move; lining-up patches that drift past half a fine voxel are listed to be judged again (added 2026-11-03) |
| 4 | score against the hold-out set, and the bar check | auto | scores of each fit and of the atlas transform; waits for 2 and 3 |
| 4b | draw and render the blind round | auto | bend against affine, on hold-out patches lining up or partly |
| **5** | **judge the final fit** | **reviewer** | blind round of bend against affine, and the decision |
| 6 | store | auto | the release record, ready for the atlas to ingest (ingesting happens there) |

## The per-pair record

One folder per pair, written only by the tool. The first four files are the part that is kept
and shared; `events.jsonl` is the internal full log the others are condensed from.

| file | holds |
|---|---|
| `pair.json` | both volumes (URL and version or ETag), voxel sizes, region, the start transform and its hash |
| `holdout.json` | the hold-out patches: coordinates, size, the measurement shown, the answer (per patch or per octant), reviewer, time, the hash of each image judged |
| `final-review.json` | the blind round (items, option order, picks, reviewer, time) and the **decision**: bend, affine or reject, its reason, and an `override` (who, why) wherever it departs from what the bar says |
| `fit/` | the final transform in the atlas format; the patches it used with their measurements and weights; the model's parameters; the scores against the bar, the atlas transform's too; the bar version; the code commit |
| `events.jsonl` | every answer and every job, append-only |

**The reviewer** is recorded by handle and email, from config, on every answer and decision.

**The bar** (agreed 2026-10-22) is config with a version, not code: on at least 20 hold-out
patches judged "lines up", median ≤ 2 and p90 ≤ 4 fine voxels, at most 1 in 20 over 5; a bend
is kept only if it wins its blind round, and the reviewer may keep a barely visible bend out.
The acceptance result is stored with the fit, and can be overridden with a reason.

## Driving it: one command, explicit state

- **State comes from the artifacts, like make.** Each stage records its inputs (with hashes),
  parameters and code commit. A stage is *done* when its output exists and its inputs have
  not changed, *stale* when they have, *waiting* when a person is next, *running* or
  *failed* (with the log).
- **`volreg run <pair>`** advances whatever can advance, starts jobs in the background, and
  stops at a human stage with the link to review. Running it again is always safe.
- **`volreg status`** shows every pair: stage, running jobs, what waits for the reviewer, the
  last error. The UI shows the same, and can do what the command does.
- **Redo is explicit:** `volreg redo <pair> <stage>` marks that stage and everything after it
  stale; `volreg more <pair> --holdout N | --fit N` draws more patches. Answers already given
  stay valid.

## Where it runs

**One container on Kubernetes**, holding the UI, the API and the worker:

- **Stateless.** The record and the artifacts live in an S3 prefix (versioned, which is also
  the backup), behind a storage layer that also takes a local folder, so the same code runs
  on a laptop. Blocks are cached in an `emptyDir` and synced out with `aws s3 sync`; never
  written to a mountpoint-S3 volume.
- **Next to the data.** The volumes are in us-east-1 (`vesuvius-challenge-open-data`,
  `scrollprize-volumes`, `scrollprize-reconstruction`).
- **Scale to zero, manually first.** A Deployment at 0 or 1 replicas, reached with
  `kubectl port-forward`, so there is no public endpoint and no auth to build yet. Automatic
  scale-to-zero (for example a request-driven scaler) comes later, if the manual way chafes.
- **Review needs no compute.** Review images are rendered once, in stage 1, and stored;
  the review page only reads them and appends answers.
- **Measure before tuning.** How much of a correlation pass is download and how much compute
  has not been measured yet; the octant fit (2.5 h on PHerc Paris4 1 µm) is compute.

## Layout

```
pipeline/
  DESIGN.md  README.md  requirements.txt
  volreg/
    record/   formats and the store (local folder or S3 prefix)
    cores/    correlation, gated fit, deformable, transform fit and I/O, merged in
    stages/   one module per stage
    engine/   state, jobs, redo
    review/   the review UI and its API
    cli.py
  tests/
  deploy/     container image and Kubernetes manifests
```

## Acceptance

**Migration first.** The first thing the new code does is convert the 16 sessions of the
first runs (`.claude/volreg-refined/sessions/`) into the new record. From that record alone,
it must reproduce the scores of the 10 staged transforms
(`.claude/volreg-refined/publish-candidates/`). A difference is either explained or stored
as an override.
