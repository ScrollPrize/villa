"""volreg: drive and inspect registration records.

    python -m volreg.cli new PAIR --start TRANSFORM --manifest MANIFEST [--name KEY]
    python -m volreg.cli run PAIR                  start every stage that can run; show where it stands
    python -m volreg.cli status RECORDS_DIR        every pair, one line each
    python -m volreg.cli answer PAIR UNIT ANSWER   a hold-out answer (or "clear")
    python -m volreg.cli more PAIR --holdout N | --fit N
    python -m volreg.cli redo PAIR STAGE
    python -m volreg.cli adopt PAIR STAGE REASON   keep a result after a change to what its stage needs
    python -m volreg.cli revisit PAIR TEXT         an open question for a later round (--done N closes it)
    python -m volreg.cli check RECORD [RECORD ...]
    python -m volreg.cli migrate SESSION OUT [--staged DIR]
    python -m volreg.cli import PAIR SESSION RECORD          a first run's patches, fits and answers
    python -m volreg.cli release OUT RECORDS_DIR [...]      every pair's newest decided record

The reviewer is taken from --handle and --email, or VOLREG_HANDLE and VOLREG_EMAIL.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from . import engine
from . import migrate as migration
from . import record as rec
from . import release as releasing
from . import stages as pipeline_stages
from .cores.transform_io import read_transform_json
from .scoring import bar_check, bar_failures, score


def records(paths):
    for path in paths:
        path = Path(path)
        if (path / rec.PAIR).exists():
            yield rec.Record(path)
        else:
            yield from (rec.Record(child) for child in sorted(path.iterdir())
                        if (child / rec.PAIR).exists())


def stage_of(record):
    """Where a record stands, from its files alone."""
    final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
    held = record.read(rec.HOLDOUT)["patches"] if record.exists(rec.HOLDOUT) else []
    if not held:
        return "needs a hold-out set"
    if not final.get("decision") and final.get("status", "").startswith("open"):
        return final["status"]
    answered = sum(1 for patch in held if patch.get("answer")
                   or any(octant.get("answer") for octant in patch.get("octants") or []))
    if answered == 0:
        return f"waiting: judge the hold-out set ({len(held)} patches)"
    if not record.exists(rec.FIT):
        return f"needs a fit ({answered} of {len(held)} hold-out patches answered)"
    if final.get("decision"):
        decision = final["decision"]
        override = " (override)" if decision.get("override") else ""
        open_items = [item for item in final.get("revisit") or [] if not item.get("done")]
        revisit = f"; revisit: {open_items[0]['text']}" if open_items else ""
        return f"decided: publish {decision['publish']} of {decision['fit']}{override}{revisit}"
    return final.get("status") or "waiting: judge the final fit"


def reviewer(args):
    return rec.Reviewer(args.handle or os.environ.get("VOLREG_HANDLE"),
                        args.email or os.environ.get("VOLREG_EMAIL")).as_json()


def show(pipeline):
    states = pipeline.states()
    for name in pipeline.order:
        stage, state = pipeline.stages[name], states[name]
        line = f"  {state:8s} {name:14s} {stage.title}"
        stamp = pipeline.stamp(name)
        if state == engine.FAILED and stamp:
            line += f"  -- {stamp['error']}"
        if state == engine.RUNNING:
            line += f"  -- log: {pipeline.log_path(name)}"
        print(line)


def in_uv_build_env(executable):
    """Whether an interpreter lives in an environment `uv run --with-requirements` deletes
    on exit: under uv's cache, in `builds-v<N>`, whichever separator the path uses."""
    return re.search(r"[\\/]builds-v\d*[\\/]", str(executable)) is not None


class CannotLaunch(RuntimeError):
    """A background job could not be started with what it needs."""


def uv_executable():
    """The uv to start a job through, by absolute path: the one running this process (`uv
    run` names itself in `UV`), else the one on PATH; None if there is neither."""
    named = os.environ.get("UV")
    if named and Path(named).is_file():
        return str(Path(named).absolute())
    return shutil.which("uv")


def python():
    """How a background job starts Python. `uv run --with-requirements` runs in an
    environment it deletes when it exits, so a job started from a `volreg run` that has
    since finished would find no interpreter; such a job goes through `uv run` again, by the
    uv that ran this (which need not be on PATH), with the requirements the README's command
    gives. With no uv to be found, CannotLaunch says what to do: the base interpreter the
    environment was made from has none of the requirements."""
    if not in_uv_build_env(sys.executable):
        return [sys.executable]
    uv = uv_executable()
    if uv is None:
        raise CannotLaunch(
            "volreg runs in a temporary uv environment, which uv deletes when this command "
            "exits, and no uv was found to start background jobs through (UV is not set and "
            "uv is not on PATH). Put uv on PATH, or set UV to the uv executable, or run volreg "
            "from an environment that lasts (python -m venv, then pip install -r "
            "requirements.txt); nothing was started.")
    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    return [uv, "run", "-q", "--python", f"{sys.version_info.major}.{sys.version_info.minor}",
            "--with-requirements", str(requirements), "python"]


def new_pair(directory, start, volumes, name, by):
    """A pair that starts from a transform file (moving -> fixed, XYZ), its volumes taken
    from a manifest entry: the `*_direct` URL where there is one. Only the start's affine
    part is kept; the fit measures again whatever deformation it carried."""
    record = rec.Record(directory)
    if record.exists(rec.PAIR):
        raise FileExistsError(f"{directory} already holds a pair")
    side = {which: {"url": volumes.get(f"{which}_direct") or volumes[which],
                    "voxel_um": float(volumes[f"{which}_voxel_size_um"])}
            for which in ("fixed", "moving")}
    matrix, _fixed_points, _moving_points = read_transform_json(start)
    record.write(rec.PAIR, {
        "pair": name, **side,
        "coarse_side": "fixed" if side["fixed"]["voxel_um"] >= side["moving"]["voxel_um"] else "moving",
        "start": {"source": str(Path(start).resolve()), "matrix_xyz": matrix[:3].tolist()},
        "created": rec.now(),
    })
    record.log("pair-created", pair=name, start=str(Path(start).resolve()), by=by)
    return record


def advance(pair):
    """Start every stage of a pair that can run, each as a background `volreg job`. How a
    job starts Python is settled before the first one starts: if it cannot be, CannotLaunch
    is raised and nothing has started."""
    pipeline = pipeline_stages.pipeline(pair)
    command = []

    def launch(name):
        if not command:
            command.extend(python() + ["-m", "volreg.cli", "job", str(pair)])
        pipeline.start(name, command + [name])
    _states, started = pipeline.advance(launch)
    return pipeline, started


def run(args):
    try:
        pipeline, started = advance(args.pair)
    except CannotLaunch as error:
        sys.exit(f"{args.pair}: {error}")
    print(f"{args.pair}: started {', '.join(started) if started else 'nothing'}")
    show(pipeline)


def is_pipeline(record):
    return record.path("work").exists()


def status(args):
    for record in records(args.records):
        if is_pipeline(record):
            states = pipeline_stages.pipeline(record.directory).states()
            waiting = [name for name, state in states.items() if state in (engine.WAITING,)]
            busy = [name for name, state in states.items() if state in (engine.RUNNING,)]
            failed = [name for name, state in states.items() if state == engine.FAILED]
            parts = "  ".join(f"{label}: {', '.join(names)}" for label, names in
                              (("waiting for you", waiting), ("running", busy), ("failed", failed))
                              if names)
            print(f"{record.directory.name:18s} " + (parts or ", ".join(
                f"{name} {state}" for name, state in states.items() if state != engine.DONE)
                or "finished"))
            continue
        pair = record.read(rec.PAIR)
        line = f"{record.directory.name:18s} {pair['pair'][:52]:52s} {stage_of(record)}"
        if record.exists(rec.FIT):
            fit = record.read(rec.FIT)
            bend = fit["scores"]["bend"]
            if bend.get("lines_up"):
                line += (f"  | bend {bend['median_fine_voxels']:.2f}/{bend['p90_fine_voxels']:.2f}"
                         f" on {bend['lines_up']}")
        print(line)


def same_score(result, stored):
    """Whether a recomputed score is the stored one: the same units, and (when any line up)
    the same distances. A score with no unit lining up has no distances to compare."""
    if result.get("lines_up", 0) != stored.get("lines_up", 0):
        return False
    if "per_unit" in stored and set(stored["per_unit"]) != set(result.get("per_unit") or {}):
        return False
    if not result.get("lines_up"):
        return True
    keys = ("median_fine_voxels", "p90_fine_voxels", "worst_fine_voxels")
    return all(isinstance(stored.get(key), (int, float)) and abs(result[key] - stored[key]) < 1e-9
               for key in keys)


def check(args):
    """Check that the stored transform, its metadata and its scores are the one result the
    decision names (the fit, the variant, the transform's digest: see `release.stored_mismatch`),
    that those scores are of the hold-out answers the record holds and are what its decision
    and any override were made on (`release.scores_problem`), then recompute the published
    transform's score from the record alone and compare it with the stored one, and check the
    decision against the bar. Exit 1 on any mismatch; a record with no unit lining up is
    checked as any other, never raising."""
    failed = 0
    for record in records(args.records):
        final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
        decision = final.get("decision")
        if not decision or not record.exists(rec.FIT):
            continue
        mismatch = releasing.stored_mismatch(record, decision, legacy_too=is_pipeline(record)) or (
            decision["publish"] != "nothing" and releasing.scores_problem(record, decision))
        if mismatch or decision["publish"] == "nothing":
            failed += bool(mismatch)
            print(f"{record.directory.name:18s} {decision['publish']:6s} of {decision['fit']}  "
                  + (f"NOT WHAT IS STORED: {mismatch}" if mismatch else "nothing published"))
            continue
        pair, fit = record.read(rec.PAIR), record.read(rec.FIT)
        result = score(record.read(rec.TRANSFORM), record.read(rec.HOLDOUT),
                       pair["coarse_side"] == "fixed", fit["fine_um"])
        stored = fit["scores"][decision["publish"]]
        same = same_score(result, stored)
        missed = bar_failures(result | {"bar": bar_check(result)})
        covered = set((decision.get("override") or {}).get("bar_missed") or [])
        consistent = set(missed) <= covered
        failed += not (same and consistent)
        measured = (f"{result['median_fine_voxels']:.2f}/{result['p90_fine_voxels']:.2f} on {result['lines_up']}"
                    if result.get("lines_up") else "no unit lines up")
        print(f"{record.directory.name:18s} {decision['publish']:6s} {measured}"
              f"  {'same as stored' if same else 'DIFFERS FROM STORED'}"
              f"  bar {'passes' if not missed else 'missed: ' + ', '.join(missed)}"
              f"{'' if not missed else (' (override)' if consistent else ' NOT COVERED BY AN OVERRIDE')}")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="volreg", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("status", "where each record stands"),
                       ("check", "recompute published scores from the record alone")):
        commands.add_parser(name, help=text).add_argument("records", nargs="+")
    commands.add_parser("run", help="start what can run, show the stages").add_argument("pair")
    new = commands.add_parser("new", help="a new pair from a start transform")
    new.add_argument("pair")
    new.add_argument("--start", required=True, help="transform JSON, moving -> fixed")
    new.add_argument("--manifest", required=True, help="pairs' volumes, by pair name")
    new.add_argument("--name", help="the pair's key in the manifest (default: the start's file name)")
    job = commands.add_parser("job", help="run one stage here (what `run` starts)")
    job.add_argument("pair")
    job.add_argument("stage")
    redo = commands.add_parser("redo", help="run a stage and what depends on it again")
    redo.add_argument("pair")
    redo.add_argument("stage")
    adopt = commands.add_parser("adopt", help="keep a result after a change to what its stage needs")
    adopt.add_argument("pair")
    adopt.add_argument("stage")
    adopt.add_argument("reason")
    more = commands.add_parser("more", help="ask for more hold-out or fit patches")
    more.add_argument("pair")
    more.add_argument("--holdout", type=int, default=0)
    more.add_argument("--fit", type=int, default=0)
    revisit = commands.add_parser("revisit", help="note an open question for a later round")
    revisit.add_argument("pair")
    revisit.add_argument("text", nargs="?")
    revisit.add_argument("--done", type=int, metavar="N", help="close open question N")
    answer = commands.add_parser("answer", help="answer about a hold-out unit")
    answer.add_argument("pair")
    answer.add_argument("unit")
    answer.add_argument("answer", choices=rec.ANSWERS + ("clear",))
    migrate = commands.add_parser("migrate", help="convert a session of the first runs")
    migrate.add_argument("session")
    migrate.add_argument("out")
    migrate.add_argument("--staged")
    imported = commands.add_parser("import", help="take a first run's patches, fits and answers")
    imported.add_argument("pair")
    imported.add_argument("session")
    imported.add_argument("record")
    released = commands.add_parser("release", help="gather every pair's newest decided record")
    released.add_argument("out")
    released.add_argument("records", nargs="+")
    for command in (answer, migrate, more, new, redo, revisit):
        command.add_argument("--handle")
        command.add_argument("--email")
    args = parser.parse_args(argv)
    if args.command == "status":
        status(args)
    elif args.command == "check":
        return check(args)
    elif args.command == "run":
        run(args)
    elif args.command == "new":
        key = args.name or Path(args.start).name
        with open(args.manifest) as handle:
            volumes = json.load(handle)[key if key.endswith(".json") else key + ".json"]
        new_pair(args.pair, args.start, volumes, key.removesuffix(".json"), reviewer(args))
        run(args)
    elif args.command == "job":
        # A job that finished starts what its result made ready, so a pair moves on by
        # itself until a person is next. Two jobs finishing at once may both try to start
        # the same stage; its lock lets only one run.
        try:
            pipeline = pipeline_stages.pipeline(args.pair)
        except Exception as error:   # noqa: BLE001 - recorded as the stage's failure
            # The pair's settings cannot be read (held past the wait for a replace, say): the
            # stage fails as a job does, under its lock, instead of being left ready with no
            # job on it.
            engine.Pipeline(args.pair, pipeline_stages.stages()).fail_unstarted(args.stage, error)
            print(f"{args.stage}: failed before it began: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        code = engine.main_job(pipeline, args.stage)
        if code == 0:
            try:
                advance(args.pair)
            except CannotLaunch as error:
                print(f"{args.pair}: {error}", file=sys.stderr)
        return code
    elif args.command == "import":
        pipeline = pipeline_stages.pipeline(args.pair)
        with pipeline_stages.session_lock(pipeline):
            print(migration.import_session(args.session, args.record, args.pair))
        try:
            advance(args.pair)
        except CannotLaunch as error:
            sys.exit(f"{args.pair}: imported, but {error}")
    elif args.command == "release":
        index = releasing.release(args.out, args.records)
        for pair in index["pairs"]:
            missing = f"  (not recorded: {', '.join(pair['not_recorded'])})" if pair["not_recorded"] else ""
            print(f"{pair['pair'][:52]:52s} {pair['decision']['publish']:6s} from {pair['source']}{missing}")
        for item in index["left_out"]:
            print(f"left out: {item['source']}: {item['why']}")
    elif args.command == "adopt":
        changed = pipeline_stages.pipeline(args.pair).adopt_needs(args.stage, args.reason)
        print(f"{args.stage}: {'now rests on its new needs' if changed else 'already up to date'}")
    elif args.command == "redo":
        affected = pipeline_stages.pipeline(args.pair).redo(args.stage)
        print(f"will run again: {', '.join(affected)}")
    elif args.command == "more":
        try:
            params = pipeline_stages.more(rec.Record(args.pair), args.holdout, args.fit, reviewer(args))
        except ValueError as error:
            parser.error(str(error))
        print(params)
    elif args.command == "revisit":
        record = rec.Record(args.pair)
        by = reviewer(args)
        # Under the lock the views are rewritten under, so a blind round or a decision
        # written meanwhile is not lost.
        with record.lock():
            final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
            final.pop("schema", None)
            items = final.setdefault("revisit", [])
            if args.done is not None:
                items[args.done - 1]["done"] = {"at": rec.now(), "by": by}
            else:
                items.append({"text": args.text, "at": rec.now(), "by": by, "done": None})
            record.write(rec.FINAL_REVIEW, final)
        if args.done is not None:
            record.log("revisit-done", item=args.done, by=by)
        else:
            record.log("revisit", text=args.text, by=by)
        for index, item in enumerate(items, 1):
            print(f"{index}. {'done' if item['done'] else 'open'}: {item['text']}")
    elif args.command == "answer":
        rec.Record(args.pair).answer(args.unit, None if args.answer == "clear" else args.answer,
                                     reviewer(args))
    else:
        migration.migrate(args.session, args.out, reviewer(args), args.staged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
