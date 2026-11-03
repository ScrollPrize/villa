"""volreg: drive and inspect registration records.

    python -m volreg.cli run PAIR                  start every stage that can run; show where it stands
    python -m volreg.cli status RECORDS_DIR        every pair, one line each
    python -m volreg.cli answer PAIR UNIT ANSWER   a hold-out answer (or "clear")
    python -m volreg.cli more PAIR --holdout N | --fit N
    python -m volreg.cli redo PAIR STAGE
    python -m volreg.cli revisit PAIR TEXT         an open question for a later round (--done N closes it)
    python -m volreg.cli check RECORD [RECORD ...]
    python -m volreg.cli migrate SESSION OUT [--staged DIR]

The reviewer is taken from --handle and --email, or VOLREG_HANDLE and VOLREG_EMAIL.
"""

import argparse
import os
import sys
from pathlib import Path

from . import engine
from . import migrate as migration
from . import record as rec
from . import stages as pipeline_stages
from .scoring import bar_check, score


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


def advance(pair):
    """Start every stage of a pair that can run, each as a background `volreg job`."""
    pipeline = pipeline_stages.pipeline(pair)
    command = [sys.executable, "-m", "volreg.cli", "job", str(pair)]
    _states, started = pipeline.advance(lambda name: pipeline.start(name, command + [name]))
    return pipeline, started


def run(args):
    pipeline, started = advance(args.pair)
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
            print(f"{record.directory.name:18s} " + "  ".join(
                f"{label}: {', '.join(names)}" for label, names in
                (("waiting for you", waiting), ("running", busy), ("failed", failed)) if names)
                or f"{record.directory.name:18s} " + ", ".join(f"{n} {s}" for n, s in states.items()))
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


def check(args):
    """Recompute the published transform's score from the record alone, and compare it with
    the stored one; check the decision against the bar. Exit 1 on any mismatch."""
    failed = 0
    for record in records(args.records):
        final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
        decision = final.get("decision")
        if not decision or not record.exists(rec.FIT):
            continue
        pair, fit = record.read(rec.PAIR), record.read(rec.FIT)
        result = score(record.read(rec.TRANSFORM), record.read(rec.HOLDOUT),
                       pair["coarse_side"] == "fixed", fit["fine_um"])
        stored = fit["scores"][decision["publish"]]
        same = all(abs(result[key] - stored[key]) < 1e-9 for key in
                   ("median_fine_voxels", "p90_fine_voxels", "worst_fine_voxels")) \
            and result["lines_up"] == stored["lines_up"]
        passes = bar_check(result)["passes"]
        consistent = passes or bool(decision.get("override"))
        failed += not (same and consistent)
        print(f"{record.directory.name:18s} {decision['publish']:6s} "
              f"{result['median_fine_voxels']:.2f}/{result['p90_fine_voxels']:.2f} on {result['lines_up']}"
              f"  {'same as stored' if same else 'DIFFERS FROM STORED'}"
              f"  bar {'passes' if passes else 'missed'}"
              f"{'' if passes else (' (override)' if decision.get('override') else ' NO OVERRIDE')}")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="volreg", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("status", "where each record stands"),
                       ("check", "recompute published scores from the record alone")):
        commands.add_parser(name, help=text).add_argument("records", nargs="+")
    commands.add_parser("run", help="start what can run, show the stages").add_argument("pair")
    job = commands.add_parser("job", help="run one stage here (what `run` starts)")
    job.add_argument("pair")
    job.add_argument("stage")
    redo = commands.add_parser("redo", help="run a stage and what depends on it again")
    redo.add_argument("pair")
    redo.add_argument("stage")
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
    for command in (answer, migrate, more, redo, revisit):
        command.add_argument("--handle")
        command.add_argument("--email")
    args = parser.parse_args(argv)
    if args.command == "status":
        status(args)
    elif args.command == "check":
        return check(args)
    elif args.command == "run":
        run(args)
    elif args.command == "job":
        # A job that finished starts what its result made ready, so a pair moves on by
        # itself until a person is next. Two jobs finishing at once may both try to start
        # the same stage; its lock lets only one run.
        status = engine.main_job(pipeline_stages.pipeline(args.pair), args.stage)
        if status == 0:
            advance(args.pair)
        return status
    elif args.command == "redo":
        affected = pipeline_stages.pipeline(args.pair).redo(args.stage)
        print(f"will run again: {', '.join(affected)}")
    elif args.command == "more":
        record = rec.Record(args.pair)
        params = record.read(pipeline_stages.PARAMS) if record.exists(pipeline_stages.PARAMS) else {}
        params.pop("schema", None)
        defaults = {stage.name: stage.params for stage in pipeline_stages.stages()}
        for stage, extra in (("holdout", args.holdout), ("fit", args.fit)):
            if extra:
                current = params.get(stage, {}).get("patches", defaults[stage]["patches"])
                params.setdefault(stage, {})["patches"] = current + extra
        record.write(pipeline_stages.PARAMS, params)
        record.log("more", holdout=args.holdout, fit=args.fit, by=reviewer(args))
        print(params)
    elif args.command == "revisit":
        record = rec.Record(args.pair)
        final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
        final.pop("schema", None)
        items = final.setdefault("revisit", [])
        if args.done is not None:
            items[args.done - 1]["done"] = {"at": rec.now(), "by": reviewer(args)}
            record.log("revisit-done", item=args.done, by=reviewer(args))
        else:
            items.append({"text": args.text, "at": rec.now(), "by": reviewer(args), "done": None})
            record.log("revisit", text=args.text, by=reviewer(args))
        record.write(rec.FINAL_REVIEW, final)
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
