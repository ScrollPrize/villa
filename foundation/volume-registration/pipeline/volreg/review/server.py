"""The review server: every pair under one folder, the two things a person does, and the
buttons that drive the rest.

It never reads a volume. What it shows comes from the record and the pictures the stages
stored; what a person does is appended to the record's event log, under the reviewer the
server was started for. Jobs it starts run as `volreg job`, as `volreg run` starts them.

    python -m volreg.review.server PAIRS_DIR [--records DIR ...] [--port 8790] [--host 127.0.0.1]
        [--handle H --email E]       (or VOLREG_HANDLE, VOLREG_EMAIL)

`--records` adds folders of records to the fits page only, read-only: the first runs'
migrated records, say, which have no pipeline to drive.

    GET  /                                    every pair and its stages
    GET  /fits                                every pair's fit: parameters, state, outcome
    GET  /api/fits
    GET  /pairs/<pair>/holdout[?unit=octants] judge the hold-out set
    GET  /pairs/<pair>/blind                  the blind round, then the decision
    GET  /api/pairs                           the pairs, their stages and what waits
    GET  /api/pairs/<pair>/holdout[?unit=octants&reveal=1]
    GET  /api/pairs/<pair>/blind
    GET  /img/<pair>/<path>                   a stored picture, under work/images
    POST /api/pairs/<pair>/run                start every stage that can run
    POST /api/pairs/<pair>/redo               {"stage": ...}
    POST /api/pairs/<pair>/more               {"holdout": n, "fit": n}
    POST /api/pairs/<pair>/params             {"stage": ..., "values": {setting: n}}
    POST /api/pairs/<pair>/answer             {"unit": ..., "answer": ... or null}
    POST /api/pairs/<pair>/note               {"patch": ..., "text": ...}
    POST /api/pairs/<pair>/pick               {"round": ..., "index": i, "choice": i or "unsure"/"none"}
    POST /api/pairs/<pair>/decide             {"publish": bend|affine|nothing, "reason": ...}
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import cli
from .. import engine
from .. import record as rec
from .. import stages as pipeline_stages

HERE = Path(__file__).resolve().parent
PAGES = {"": "index.html", "holdout": "holdout.html", "blind": "blind.html"}
# How a fit is shown on the fits page: the session's own record of it, these keys only.
FIT_HISTORY_KEYS = ("id", "parent", "at", "model", "held_out_rms", "robust", "unit")
TEST_STRETCHES = 5
# The settings the pair page offers, per stage: what a person may want to change between
# runs. Everything else stays as `stages.py` sets it, and `params.json` can still hold it.
EDITABLE = {"holdout": {"patches": "hold-out patches"},
            "fit": {"patches": "fit patches", "passes": "passes"},
            "blind": {"count": "patches in a round"}}


class Reviews:
    """The pairs under a folder, and what the pages ask of them."""

    def __init__(self, root, reviewer, auto_advance=True, records=()):
        self.root = Path(root).resolve()
        self.records = [Path(folder).resolve() for folder in records]
        self.reviewer = reviewer
        # Whether answers and decisions start the jobs they make ready (off in tests).
        self.auto_advance = auto_advance

    def pair_dir(self, name):
        path = (self.root / name).resolve()
        if path.parent != self.root or not (path / rec.PAIR).exists():
            raise KeyError(name)
        return path

    def record(self, name):
        return rec.Record(self.pair_dir(name))

    def pipeline(self, name):
        return pipeline_stages.pipeline(self.pair_dir(name))

    # ---------------------------------------------------------------------- the index

    def pairs(self):
        found = []
        for path in sorted(self.root.iterdir()):
            if not (path / rec.PAIR).exists() or not (path / "work").exists():
                continue
            pipeline = pipeline_stages.pipeline(path)
            states = pipeline.states()
            entry = {"name": path.name, "pair": pipeline.record.read(rec.PAIR)["pair"],
                     "stages": [self.stage_entry(pipeline, name, states[name])
                                for name in pipeline.order]}
            if pipeline.record.exists(rec.HOLDOUT):
                entry["holdout"] = self.progress(pipeline.record.read(rec.HOLDOUT), False)
                entry["target"] = pipeline.stage_params("judge-holdout")["min_lines_up"]
            if pipeline.record.exists(rec.FINAL_REVIEW):
                final = pipeline.record.read(rec.FINAL_REVIEW)
                entry["decision"] = final.get("decision")
                entry["revisit"] = [item["text"] for item in final.get("revisit") or []
                                    if not item.get("done")]
            found.append(entry)
        return {"pairs": found, "reviewer": self.reviewer}

    @staticmethod
    def stage_entry(pipeline, name, state):
        stage, stamp = pipeline.stages[name], pipeline.stamp(name) or {}
        params = pipeline.stage_params(name)
        return {"name": name, "title": stage.title, "human": stage.human, "state": state,
                "error": stamp.get("error") if state == engine.FAILED else None,
                "finished": stamp.get("finished"),
                "settings": [{"key": key, "label": label, "value": params.get(key)}
                             for key, label in EDITABLE.get(name, {}).items()]}

    def set_params(self, name, stage, values):
        """Change a stage's settings; the stage and what follows it run again."""
        allowed = EDITABLE.get(stage, {})
        if not values or set(values) - set(allowed):
            raise ValueError(f"{stage} has no setting {sorted(set(values) - set(allowed))}")
        clean = {}
        for key, value in values.items():
            value = int(value)
            if value < 1:
                raise ValueError(f"{allowed[key]} must be at least 1")
            clean[key] = value
        record = self.record(name)
        params = record.read(pipeline_stages.PARAMS) if record.exists(pipeline_stages.PARAMS) else {}
        params.pop("schema", None)
        params.setdefault(stage, {}).update(clean)
        record.write(pipeline_stages.PARAMS, params)
        record.log("params", stage=stage, values=clean, by=self.reviewer)
        self.advance(name)
        return params

    def run(self, name):
        _pipeline, started = cli.advance(self.pair_dir(name))
        return {"started": started}

    def advance(self, name):
        if self.auto_advance:
            cli.advance(self.pair_dir(name))

    def redo(self, name, stage):
        affected = self.pipeline(name).redo(stage)
        self.advance(name)
        return {"affected": affected}

    def more(self, name, holdout, fit):
        record = self.record(name)
        params = record.read(pipeline_stages.PARAMS) if record.exists(pipeline_stages.PARAMS) else {}
        params.pop("schema", None)
        defaults = {stage.name: stage.params for stage in pipeline_stages.stages()}
        for stage, extra in (("holdout", holdout), ("fit", fit)):
            if extra:
                current = params.get(stage, {}).get("patches", defaults[stage]["patches"])
                params.setdefault(stage, {})["patches"] = current + int(extra)
        record.write(pipeline_stages.PARAMS, params)
        record.log("more", holdout=holdout, fit=fit, by=self.reviewer)
        return params

    # ------------------------------------------------------------------ the hold-out set

    @staticmethod
    def units(held, octants):
        """The hold-out set as it is judged: patches, or the octants that measured well."""
        found = []
        for patch in held["patches"]:
            note = (patch.get("note") or {}).get("text")
            if not octants:
                found.append({"id": patch["id"], "patch": patch["id"], "stretch": patch["stretch"],
                              "origin_zyx": patch["origin_zyx"], "size": held["patch_coarse_voxels"],
                              "answer": patch.get("answer"), "note": note})
                continue
            for octant in patch.get("octants") or []:
                origin = [a + b for a, b in zip(patch["origin_zyx"], octant["corner_zyx"])]
                found.append({"id": f"{patch['id']}.{octant['index']}", "patch": patch["id"],
                              "stretch": patch["stretch"], "origin_zyx": origin,
                              "size": held["patch_coarse_voxels"] // 2,
                              "answer": octant.get("answer"), "note": note})
        return found

    def progress(self, held, octants):
        """Per stretch: how many line up, are answered, are left; and the answers' counts."""
        stretches = [{"lines_up": 0, "answered": 0, "left": 0} for _ in range(TEST_STRETCHES)]
        counts = dict.fromkeys(rec.ANSWERS, 0)
        for unit in self.units(held, octants):
            into = stretches[unit["stretch"]]
            if not unit["answer"]:
                into["left"] += 1
                continue
            answer = unit["answer"]["answer"]
            into["answered"] += 1
            into["lines_up"] += answer == rec.LINES_UP
            counts[answer] += 1
        return {"stretches": stretches, "answers": counts, "lines_up": counts[rec.LINES_UP]}

    def holdout(self, name, octants=False, reveal=False):
        pipeline = self.pipeline(name)
        held = pipeline.record.read(rec.HOLDOUT)
        units = self.units(held, octants)
        progress = self.progress(held, octants)
        target = pipeline.stage_params("judge-holdout")["min_lines_up"]
        waiting = [unit for unit in units if not unit["answer"]]
        stretches = progress["stretches"]
        following = min(waiting, key=lambda unit: (stretches[unit["stretch"]]["lines_up"],
                                                   unit["stretch"], unit["id"]))["id"] if waiting else None
        done = progress["lines_up"] >= target or not waiting
        scores = pipeline.work("scores.json")
        return {"pair": pipeline.record.read(rec.PAIR)["pair"], "octants": octants,
                "answers": list(rec.ANSWERS), "units": units, "progress": progress,
                "target": target, "next": following, "done": done,
                "scores": json.loads(scores.read_text()) if scores.exists() and (done or reveal)
                else None,
                "score_stale": pipeline.states()["score"] != engine.DONE}

    def answer(self, name, unit, answer):
        """Record an answer, then start what it made ready: scoring, once enough line up."""
        self.record(name).answer(unit, answer, self.reviewer)
        self.advance(name)

    def note(self, name, patch, text):
        self.record(name).note(patch, text, self.reviewer)

    # ---------------------------------------------------------------- the final review

    def blind(self, name):
        """The newest round as a page may see it: which option is which, the tally, the
        scores and the bar only once every item is answered."""
        pipeline = self.pipeline(name)
        final = pipeline.record.read(rec.FINAL_REVIEW) if pipeline.record.exists(rec.FINAL_REVIEW) else {}
        rounds = final.get("rounds") or []
        if not rounds:
            return {"round": None, "decision": final.get("decision")}
        round_ = rounds[-1]
        done, tally = tally_of(round_)
        scores = pipeline.work("scores.json")
        return {
            "pair": pipeline.record.read(rec.PAIR)["pair"], "done": done,
            "round": {"id": round_["id"], "candidates": round_["candidates"] if done else None,
                      "items": [{"patch": item["patch"], "options": len(item["options"]),
                                 "choice": item["choice"]}
                                | ({"revealed": item["options"]} if done else {})
                                for item in round_["items"]]},
            "tally": tally if done else None,
            "scores": json.loads(scores.read_text()) if done and scores.exists() else None,
            "decision": final.get("decision"),
            "revisit": [item for item in final.get("revisit") or [] if not item.get("done")],
        }

    # ------------------------------------------------------------------- the fits page

    def fits(self):
        """One row per pair, the pipeline's and the read-only records alike: the fit it
        decided on (or, before a decision, the one it is scoring), with its parameters,
        where the pair stands, the scores against the hold-out set, the blind tally and the
        decision; for a pipeline pair also every fit its session made on the way."""
        rows = []
        folders = [(self.root, True)] + [(folder, False) for folder in self.records]
        for folder, driven in folders:
            for path in sorted(folder.iterdir()):
                if (path / rec.PAIR).exists() and (not driven or (path / "work").exists()):
                    rows.append(fit_row(path, driven, folder.name))
        return {"fits": rows}

    def pick(self, name, round_id, index, choice):
        self.record(name).blind_pick(round_id, int(index), choice, self.reviewer)

    def decide(self, name, publish, reason):
        """The decision, checked against the bar: publishing a variant that misses it is an
        override, which needs a reason and names what it missed."""
        pipeline = self.pipeline(name)
        final = pipeline.record.read(rec.FINAL_REVIEW)
        rounds = final.get("rounds") or []
        if not rounds or any(item["choice"] is None for item in rounds[-1]["items"]):
            raise ValueError("finish the blind round first")
        fit = json.loads(pipeline.work("fit", "fit.json").read_text())["fit"]
        scores = json.loads(pipeline.work("scores.json").read_text())["scores"]
        override = None
        if publish in ("bend", "affine") and not scores[publish]["bar"]["passes"]:
            if not (reason or "").strip():
                raise ValueError(f"the {publish} misses the bar: give a reason to publish it anyway")
            missed = [part for part, ok in scores[publish]["bar"]["checks"].items() if not ok]
            override = {"by": self.reviewer, "reason": reason, "bar_missed": missed}
        pipeline.record.decide({"fit": fit, "publish": publish, "reason": reason or "",
                                "override": override}, self.reviewer)
        self.advance(name)

    # ---------------------------------------------------------------------- pictures

    def image(self, name, path):
        base = (self.pair_dir(name) / "work" / "images").resolve()
        target = (base / path).resolve()
        if base not in target.parents or not target.is_file():
            raise KeyError(path)
        return target.read_bytes(), ("application/json" if target.suffix == ".json" else "image/png")


def tally_of(round_):
    """Whether every item of a blind round is answered, and how often each candidate (and
    "unsure", "none") was picked."""
    done = all(item["choice"] is not None for item in round_["items"])
    tally = {candidate: 0 for candidate in round_["candidates"]} | {"unsure": 0, "none": 0}
    for item in round_["items"]:
        if isinstance(item["choice"], int):
            for candidate in item["options"][item["choice"]]:
                tally[candidate] += 1
        elif item["choice"] in tally:
            tally[item["choice"]] += 1
    return done, tally


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def fit_row(path, driven, folder):
    record = rec.Record(path)
    pair = record.read(rec.PAIR)
    final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
    stored = read_json(record.path(rec.FIT))
    row = {"name": path.name, "folder": folder, "pair": pair["pair"], "driven": driven,
           "source": ("pipeline, imported" if any(event["kind"] == "session-imported"
                                                  for event in record.events())
                      else "pipeline") if driven else "first runs",
           "fine_um": pair[("moving" if pair["coarse_side"] == "fixed" else "fixed")]["voxel_um"],
           "coarse_um": pair[pair["coarse_side"]]["voxel_um"],
           "decision": final.get("decision"), "params": None, "stages": None, "history": None}
    fit, scores = stored, stored
    if driven:
        pipeline = pipeline_stages.pipeline(path)
        states = pipeline.states()
        row["stages"] = [{"name": name, "state": states[name]} for name in pipeline.order]
        row["params"] = {name: pipeline.stage_params(name) for name in ("holdout", "fit", "blind")}
        if not (final.get("decision") and stored):
            # Not stored yet: the fit the pipeline made and is scoring, if any.
            fit, scores = read_json(pipeline.work("fit", "fit.json")), read_json(pipeline.work("scores.json"))
        session = read_json(pipeline.work("session", "session.json"))
        if session:
            row["history"] = [{key: entry.get(key) for key in FIT_HISTORY_KEYS}
                              | {"points": len(entry.get("used") or []),
                                 "imported": bool(entry.get("imported_from")),
                                 "bend": bool(entry.get("bend"))}
                              for entry in session["fits"]]
            row["current"] = session.get("current")
        row["state"] = next((f"{entry['name']}: {entry['state']}" for entry in row["stages"]
                             if entry["state"] != engine.DONE), "finished")
    else:
        row["state"] = cli.stage_of(record)
    if fit:
        row["fit"] = {"id": fit["fit"], "model": fit["model"], "robust": fit.get("robust"),
                      "hold_out": fit.get("hold_out"), "unit": fit.get("unit"),
                      "points": len(fit["points"]["ids"]), "bend": fit.get("bend")}
    if scores and scores.get("scores"):
        row["scores"] = {variant: {key: value.get(key) for key in
                                   ("lines_up", "median_fine_voxels", "p90_fine_voxels",
                                    "worst_fine_voxels", "far", "median_um", "p90_um")}
                         | {"passes": (value.get("bar") or {}).get("passes")}
                         for variant, value in scores["scores"].items()}
        row["bar"] = scores.get("bar")
        row["drift"] = scores.get("drift")
    rounds = final.get("rounds") or []
    if rounds:
        done, tally = tally_of(rounds[-1])
        row["blind"] = {"round": rounds[-1]["id"], "done": done, "items": len(rounds[-1]["items"]),
                        "tally": tally if done else None}
    row["revisit"] = [item["text"] for item in final.get("revisit") or [] if not item.get("done")]
    row["status"] = final.get("status")
    return row


def handler_for(reviews):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - the stdlib's own name
            pass

        def send(self, body, kind, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send(json.dumps(value).encode(), "application/json", status)

        def do_GET(self):  # noqa: N802 - the stdlib's name
            url = urlparse(self.path)
            parts = [part for part in url.path.split("/") if part]
            query = parse_qs(url.query)
            octants = query.get("unit", ["patches"])[0] == "octants"
            try:
                if not parts:
                    self.send((HERE / PAGES[""]).read_bytes(), "text/html; charset=utf-8")
                elif parts == ["common.js"]:
                    self.send((HERE / "common.js").read_bytes(), "text/javascript; charset=utf-8")
                elif parts == ["common.css"]:
                    self.send((HERE / "common.css").read_bytes(), "text/css; charset=utf-8")
                elif parts[0] == "pairs" and len(parts) == 3 and parts[2] in PAGES:
                    reviews.pair_dir(parts[1])
                    self.send((HERE / PAGES[parts[2]]).read_bytes(), "text/html; charset=utf-8")
                elif parts == ["fits"]:
                    self.send((HERE / "fits.html").read_bytes(), "text/html; charset=utf-8")
                elif parts == ["api", "pairs"]:
                    self.send_json(reviews.pairs())
                elif parts == ["api", "fits"]:
                    self.send_json(reviews.fits())
                elif parts[:2] == ["api", "pairs"] and len(parts) == 4 and parts[3] == "holdout":
                    self.send_json(reviews.holdout(parts[2], octants,
                                                   query.get("reveal", ["0"])[0] == "1"))
                elif parts[:2] == ["api", "pairs"] and len(parts) == 4 and parts[3] == "blind":
                    self.send_json(reviews.blind(parts[2]))
                elif parts[0] == "img" and len(parts) >= 3:
                    body, kind = reviews.image(parts[1], "/".join(parts[2:]))
                    self.send(body, kind)
                else:
                    self.send_json({"error": "not found"}, 404)
            except (KeyError, IndexError, FileNotFoundError, StopIteration) as error:
                self.send_json({"error": f"not found: {error}"}, 404)
            except Exception as error:  # noqa: BLE001 - report it to the page
                self.send_json({"error": str(error)}, 500)

        def do_POST(self):  # noqa: N802 - the stdlib's name
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            parts = [part for part in urlparse(self.path).path.split("/") if part]
            try:
                if parts[:2] != ["api", "pairs"] or len(parts) != 4:
                    raise KeyError(self.path)
                name, action = parts[2], parts[3]
                if action == "run":
                    self.send_json(reviews.run(name))
                elif action == "redo":
                    self.send_json(reviews.redo(name, body["stage"]))
                elif action == "more":
                    self.send_json(reviews.more(name, body.get("holdout", 0), body.get("fit", 0)))
                elif action == "params":
                    self.send_json(reviews.set_params(name, body["stage"], body["values"]))
                elif action == "answer":
                    reviews.answer(name, body["unit"], body.get("answer"))
                    self.send_json({"ok": True})
                elif action == "note":
                    reviews.note(name, body["patch"], body.get("text"))
                    self.send_json({"ok": True})
                elif action == "pick":
                    reviews.pick(name, body["round"], body["index"], body["choice"])
                    self.send_json({"ok": True})
                elif action == "decide":
                    reviews.decide(name, body["publish"], body.get("reason"))
                    self.send_json({"ok": True})
                else:
                    raise KeyError(action)
            except KeyError as error:
                self.send_json({"error": f"not found: {error}"}, 404)
            except (IndexError, ValueError, StopIteration) as error:
                self.send_json({"error": str(error)}, 400)
            except Exception as error:  # noqa: BLE001 - report it to the page
                self.send_json({"error": str(error)}, 500)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pairs")
    parser.add_argument("--records", nargs="*", default=[])
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--handle", default=os.environ.get("VOLREG_HANDLE"))
    parser.add_argument("--email", default=os.environ.get("VOLREG_EMAIL"))
    args = parser.parse_args(argv)
    reviews = Reviews(args.pairs, rec.Reviewer(args.handle, args.email).as_json(),
                      records=args.records)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(reviews))
    print(f"reviewing {args.pairs} as {args.handle} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
