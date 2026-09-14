"""Cached native history sweep, with optional manager-gated coverage extension."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import sys

from lasagna.inference_provenance import atomic_write, json_digest
from .cached_trace_profile import PublishedStore, extend_cached_replay
from .live_evaluate import coordinate_native, implementation_identity


def variants(controls_only=False, weight=.25):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("sweep weight must be finite and nonnegative")
    yield dict(name="off", weight=0, spacing=32, window=256, count=8, falloff="uniform")
    yield dict(name="original", weight=.25, spacing=32, window=256, count=8, falloff="uniform")
    if controls_only:
        return
    for spacing in (8, 16, 32):
        for window in (32, 64, 128, 256):
            for falloff in ("uniform", "linear", "gaussian"):
                yield dict(name=f"s{spacing}-w{window}-{falloff}", weight=weight,
                           spacing=spacing, window=window, count=64, falloff=falloff)


def replay_hash(output):
    return json_digest({p.parent.name: json.loads(p.read_text())["replay"]
                       for p in sorted((Path(output) / "cases").glob("*/trace.json"))})


def search_weight(execute, *, history=None, start_weight=.25):
    """Strict neighbor descent on 0.05 ticks; ties retain the current weight."""
    history = dict(history or dict(spacing=8, window=256, count=64, falloff="gaussian"))
    if not math.isfinite(start_weight) or start_weight < 0 or abs(start_weight * 20 - round(start_weight * 20)) > 1e-8:
        raise ValueError("start_weight must be a nonnegative multiple of .05")
    scores = {}
    def evaluate(tick):
        if tick not in scores:
            row = execute(dict(history, name=f"weight-{tick * 5:03d}", weight=tick / 20))
            if not row["completed"]:
                raise RuntimeError("incomplete weight replay cannot establish a local optimum")
            scores[tick] = row
        return scores[tick]
    current = round(start_weight * 20)
    while True:
        center = evaluate(current)
        neighbors = [t for t in (current - 1, current + 1) if t >= 0]
        for tick in neighbors:
            evaluate(tick)
        best = min([current] + neighbors, key=lambda t: scores[t]["failures"])
        if scores[best]["failures"] >= center["failures"]:
            return center
        current = best


def refine_families(execute, families, on_result=None):
    """Finite coordinate descent; each family starts from its own seed, not global best."""
    cache, winners = {}, []
    for family in families:
        axes = family['axes']
        if not axes or any(not isinstance(v,list) or not v for v in axes.values()):
            raise ValueError('refinement requires nonempty finite axes')
        configured_variants({'candidates':[family['seed']]})
        if family.get('neighbor_weight',False):
            weights=[family['seed']['weight']]+axes.get('weight',[])
            if any(not isinstance(w,(int,float)) or not math.isfinite(w) or abs(w*20-round(w*20))>1e-8 for w in weights):
                raise ValueError('neighbor refinement requires .05 weight ticks')
        for axis, values in axes.items():
            for value in values:
                configured_variants({'candidates':[dict(family['seed'],**{axis:value})]})
    def evaluate(parameters):
        validated = configured_variants({'candidates':[parameters]})[-1]
        validated.pop('name')
        key = json.dumps(validated, sort_keys=True)
        if key not in cache:
            row = execute(dict(validated,name=f'refine-{len(cache):04d}'))
            if not row['completed']:
                raise RuntimeError('incomplete refinement cannot establish a local optimum')
            cache[key] = row
        return cache[key]
    for family in families:
        current = dict(family['seed'])
        print('refinement start '+family['label']+' '+json.dumps(current,sort_keys=True),flush=True)
        axes = family['axes']
        while True:
            previous = evaluate(current)
            for axis, values in axes.items():
                center = evaluate(current)
                candidates = [(current,center)] + [(dict(current,**{axis:v}),evaluate(dict(current,**{axis:v}))) for v in values]
                parameters,best = min(candidates,key=lambda pair:pair[1]['failures'])
                if best['failures'] < center['failures']:
                    current = dict(parameters)
            if family.get('neighbor_weight',False):
                history = {k:v for k,v in current.items() if k != 'weight'}
                refined = search_weight(lambda row:evaluate({k:v for k,v in row.items() if k != 'name'}),
                                        history=history,start_weight=current['weight'])
                current['weight'] = refined['weight']
            if evaluate(current)['failures'] >= previous['failures']:
                break
        winners.append(dict(label=family['label'],axes=axes,neighbor_weight=family.get('neighbor_weight',False),parameters=current,result=evaluate(current)))
        if on_result is not None:
            on_result(winners[-1])
        print('refinement complete '+family['label']+' '+json.dumps(dict(parameters=current,failures=evaluate(current)['failures']),sort_keys=True),flush=True)
    return winners


def configured_variants(settings):
    custom = settings.get("candidates")
    if custom is None:
        return list(variants(settings.get("controls_only", False) or settings.get("weight_search", False) or bool(settings.get('refine')),
                             settings.get("grid_weight", .25)))
    if settings.get("weight_search", False) or not isinstance(custom, list) or not custom:
        raise ValueError("candidates must be nonempty and cannot accompany weight_search")
    rows = list(variants(True))
    for i, candidate in enumerate(custom):
        required = {"weight", "spacing", "window", "count", "falloff"}
        if not required <= set(candidate) or set(candidate) - required - {"mode", "half_life", "delay", "anchors", "ema_mix", "aggregation", "novelty", "touch", "temperature"}:
            raise ValueError("candidate must contain only embedding weight/spacing/window/count/falloff")
        if any(not isinstance(candidate[k], (int, float)) or not math.isfinite(candidate[k]) or candidate[k] <= 0
               for k in ("spacing", "window")):
            raise ValueError("history distances must be finite and positive")
        if type(candidate['count']) is not int or not 1 <= candidate['count'] <= 64:
            raise ValueError("history count must be integer in [1,64]")
        if not isinstance(candidate['weight'], (int, float)) or not math.isfinite(candidate['weight']) or candidate['weight'] < 0:
            raise ValueError("weight must be finite and nonnegative")
        if candidate['falloff'] not in ('uniform', 'linear', 'gaussian'):
            raise ValueError("unsupported falloff")
        mode = candidate.get('mode', 'mean')
        touch = candidate.get('touch', 'nearest')
        if touch not in ('nearest', 'hit') or (touch == 'hit' and mode != 'anchors-lru'):
            raise ValueError('invalid anchor touch policy')
        if mode not in ('mean', 'best', 'seed', 'ema', 'ema-seed', 'ema-gated', 'ema-seed-gated', 'ema-delayed', 'anchors', 'anchors-lru', 'appearance'):
            raise ValueError('unsupported reference mode')
        if mode == 'best' and candidate['falloff'] != 'uniform':
            raise ValueError('best history requires uniform falloff')
        half = candidate.get('half_life', 512)
        if not isinstance(half, (int, float)) or not math.isfinite(half) or half <= 0:
            raise ValueError('invalid EMA half life')
        delay = candidate.get('delay', 0)
        if not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0 <= delay <= 2048:
            raise ValueError('invalid EMA delay')
        if type(candidate.get('anchors', 4)) is not int or not 1 <= candidate.get('anchors', 4) <= 8:
            raise ValueError('invalid anchor count')
        mix = candidate.get('ema_mix', .5)
        if not isinstance(mix, (int, float)) or not math.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError('invalid anchor EMA mix')
        if candidate.get('aggregation','best') not in ('best','mean','normalized-mean','support-softmin'):
            raise ValueError('invalid anchor aggregation')
        if candidate.get('aggregation') == 'normalized-mean' and mode != 'anchors-lru':
            raise ValueError('normalized-mean requires anchors-lru')
        if mode == 'anchors' and candidate.get('aggregation','best') != 'best':
            raise ValueError('representative anchors require best aggregation')
        if (mode == 'appearance' and candidate.get('aggregation') == 'mean') or (mode != 'appearance' and candidate.get('aggregation') == 'support-softmin'):
            raise ValueError('support-softmin requires appearance; appearance does not use mean')
        for key, default, lower, upper in [('novelty',0,0,2),('temperature',.1,0,float('inf'))]:
            value=candidate.get(key,default)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value < lower or value > upper or (key=='temperature' and value==0):
                raise ValueError('invalid anchor '+key)
        rows.append(dict(candidate, name=f"candidate-{i:03d}"))
    return rows


def run(settings):
    scheduled = configured_variants(settings)
    root = Path(settings["pilot"])
    pilot = json.loads((root / "report.json").read_text())
    previous = {"off" if v["weight"] == 0 else "original": Path(v["output"]) for v in pilot["variants"]}
    summary = json.loads((previous["original"] / "summary.json").read_text())
    if not summary["completed"]:
        raise ValueError("completed inference required")
    base = summary["config"]
    fill_missing = settings.get("fill_missing", False)
    for source in base["sources"]:
        identity = json.loads((Path(source["fiber_manifest"]).parent / "cache_identity.json").read_text())
        if identity["identity"]["model"]["checkpoint_sha256"] != pilot["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint mismatch")
        if fill_missing and identity["identity"]["model"]["implementation"] != implementation_identity():
            raise ValueError("inference implementation changed: cannot extend frozen cache")
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint_sha256=pilot["checkpoint_sha256"], step=pilot["step"],
                  fiber=pilot["fiber"], ranking="completed only, minimum bidirectional failures",
                  inference="fill missing only under manager" if fill_missing else "disabled; missing coverage fails variant", variants=[])

    def execute(variant):
        # Off used a separate (non-embedding) cache in the pilot. Reproduce it
        # there; all positive-weight comparisons use the same signed cache.
        evaluation = deepcopy(json.loads((previous["off"] / "summary.json").read_text())["config"]
                              if variant["weight"] == 0 else base)
        evaluation.pop("live_endpoint", None)
        evaluation.update(output=str(output / variant["name"]), workers=2, render=False)
        args = evaluation["trace_args"]
        # Remove only existing embedding option/value pairs, preserving all other settings.
        cleaned, i = [], 0
        while i < len(args):
            if args[i].startswith("--embedding-"):
                i += 2
            else:
                cleaned.append(args[i]); i += 1
        for flag, key in [("weight", "weight"), ("history-spacing-base-voxels", "spacing"),
                          ("history-window-base-voxels", "window"), ("history-count", "count"),
                          ("history-falloff", "falloff")]:
            cleaned += ["--embedding-" + flag, str(variant[key])]
        evaluation["trace_args"] = cleaned
        if 'mode' in variant:
            evaluation['trace_args'] += ['--embedding-reference-mode', variant['mode']]
        if 'half_life' in variant:
            evaluation['trace_args'] += ['--embedding-ema-half-life-base-voxels', str(variant['half_life'])]
        if 'delay' in variant:
            evaluation['trace_args'] += ['--embedding-ema-delay-base-voxels', str(variant['delay'])]
        for key, flag in [('anchors', 'anchor-count'), ('ema_mix', 'anchor-ema-mix'), ('aggregation','anchor-aggregation'), ('novelty','anchor-novelty'), ('touch','anchor-touch'), ('temperature','anchor-temperature')]:
            if key in variant:
                evaluation['trace_args'] += ['--embedding-'+flag, str(variant[key])]
        atomic_write(output / (variant["name"] + ".json"), evaluation)
        stores, mapping, channels = {}, {}, {}
        for source in evaluation["sources"]:
            key = source["fiber_manifest"]
            stores[key] = PublishedStore(key)
            mapping[source["id"]] = key
            channels[key] = stores[key].channels
        row = dict(variant)
        started = time.monotonic()
        print("sweep start " + variant["name"], flush=True)
        try:
            if fill_missing and variant["name"] not in previous and variant["name"] != "repeat-best":
                extend_cached_replay(evaluation, settings["native_binary"], Path(pilot["checkpoint"]))
            else:
                coordinate_native(evaluation, Path(settings["native_binary"]), stores, mapping, channels,
                    {k: set() for k in stores}, {}, (),
                    {"cached_trace_timeout_seconds": 180}, allow_inference=False)
            inventory = json.loads((Path(evaluation["output"]) / "inference_summary.json").read_text())
            if set(inventory["cache_roots"].values()) != {str(s.root) for s in stores.values()}:
                raise RuntimeError("replay did not reuse expected cache")
            row["inferred_tiles"] = sum(s["inferred_tiles"] for s in inventory["sources"].values())
            result = json.loads((Path(evaluation["output"]) / "summary.json").read_text())
            if not result["completed"] or result["cases"] != 2:
                raise RuntimeError("incomplete bidirectional replay")
            row.update(completed=True, failures=result["failures"], summary=result,
                       replay_hash=replay_hash(evaluation["output"]))
            if variant["name"] in previous:
                row["original_replay_match"] = row["replay_hash"] == replay_hash(previous[variant["name"]])
        except RuntimeError as error:
            if "cached-only inference miss" not in str(error):
                raise
            row.update(completed=False, error=str(error))
        row["seconds"] = time.monotonic() - started
        report["variants"].append(row)
        atomic_write(output / "report.json", report)
        print("sweep result " + json.dumps({k:v for k,v in row.items() if k != "summary"}), flush=True)
        return row

    for variant in scheduled:
        row = execute(variant)
        if variant["name"] in previous and (not row["completed"] or not row["original_replay_match"]):
            raise RuntimeError("control did not reproduce original replay; inspect before interpreting sweep")
    if settings.get('refine'):
        if settings.get('weight_search'):
            raise ValueError('refine and weight_search are mutually exclusive')
        if any(not row['completed'] for row in report['variants']):
            raise RuntimeError('incomplete scheduled replay before refinement')
        report['refinements'] = []
        def family_finished(result):
            report['refinements'].append(result)
            atomic_write(output / 'report.json',report)
        refine_families(execute,settings['refine'],on_result=family_finished)
        for i, family in enumerate(report['refinements']):
            repeated = execute(dict(family['parameters'],name=f'repeat-family-{i:02d}'))
            family['repeat_identical'] = repeated.get('replay_hash') == family['result']['replay_hash']
            if not family['repeat_identical']:
                raise RuntimeError('family winner repeat differs')
        best = min(report['variants'],key=lambda v:v['failures'])
        report['ranking'] = 'per-family coordinate minima over explicitly recorded finite axes'
    elif settings.get("weight_search", False):
        best = search_weight(execute, history=settings.get('search_history'), start_weight=settings.get('start_weight', .25))
        best = min([best] + [v for v in report['variants'] if v['name'] in previous], key=lambda v: v['failures'])
        report["ranking"] = "local minimum of bidirectional failures on 0.05 weight grid; ties retain center"
    else:
        completed = [v for v in report["variants"] if v["completed"]]
        if len(completed) != len(scheduled):
            raise RuntimeError("full history grid incomplete; cannot select winner")
        best = min(completed, key=lambda v: v["failures"])
    repeated = execute({**{k:best[k] for k in ("weight", "spacing", "window", "count", "falloff", "mode", "half_life", "delay", "anchors", "ema_mix", "aggregation", "novelty", "touch", "temperature") if k in best}, "name":"repeat-best"})
    report["best"] = best["name"]
    report["repeat_identical"] = repeated.get("replay_hash") == best["replay_hash"]
    atomic_write(output / "report.json", report)
    if not report["repeat_identical"]:
        raise RuntimeError("best replay is not deterministic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("--unmanaged", action="store_true")
    parser.add_argument("--devices", default="0")
    args = parser.parse_args()
    settings = json.loads(Path(args.control).read_text())["embedding_trace_sweep"]
    if settings.get("fill_missing", False) and not args.unmanaged:
        from .managed_evaluation import launch_managed
        return launch_managed([sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep", args.control],
                              devices="0", settings=settings["gpu_manager"])
    run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
