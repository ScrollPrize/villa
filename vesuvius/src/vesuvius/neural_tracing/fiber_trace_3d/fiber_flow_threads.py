"""Benchmark HiGHS thread counts on one presolved graph, then finish the winner.

Uses SciPy's bundled HiGHS bindings (a private, version-dependent API) to retain
the presolve mapping and simplex bases. No change to the circulation objective.
"""
import json
import os
from pathlib import Path
import re
import signal
import time

import numpy as np
from scipy.optimize._highspy import _core as core

from .fiber_min_flow import certify_flow
from .fiber_min_flow_experiment import arrays, write


def checked(status):
    if status == core.HighsStatus.kError:
        raise RuntimeError('HiGHS operation failed; see solver log')


def options(h, threads, log, *, presolve=False, limit=2147483647):
    for key, value in dict(solver='simplex', threads=threads,
            parallel='on' if threads > 1 else 'off',
            simplex_strategy=3 if threads > 1 else 1,
            # HiGHS 1.12 limits simplex concurrency to eight even when the
            # scheduler has more threads. Do not request an invalid option.
            simplex_min_concurrency=min(threads, 8),
            simplex_max_concurrency=min(threads, 8),
            presolve='on' if presolve else 'off',
            simplex_iteration_limit=limit, log_to_console=False,
            log_file=str(log), random_seed=0).items():
        checked(h.setOptionValue(key, value))


def model(node_count, g):
    """Exactly the incidence matrix, costs and bounds used by solve()."""
    n = len(g['a'])
    lp = core.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = node_count
    lp.col_cost_ = -g['target'].astype(float)
    lp.col_lower_ = -g['weight']
    lp.col_upper_ = g['weight']
    lp.row_lower_ = np.zeros(node_count)
    lp.row_upper_ = np.zeros(node_count)
    lp.a_matrix_.format_ = core.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(n+1, dtype=np.int64)*2
    lp.a_matrix_.index_ = np.column_stack((g['a'], g['b'])).ravel()
    lp.a_matrix_.value_ = np.tile([-1., 1.], n)
    return lp


def log_rates(path):
    # Native timestamps omit Python model loading; adjacent rows omit the initial
    # basis factorization. Report actual logged objective progress, not just pivots.
    pattern = re.compile(r'^\s*(\d+)\s+([-+\d.eE]+)\s+Pr:.*?\s([\d.]+)s\s*$')
    rows = []
    for line in Path(path).read_text().splitlines():
        m = pattern.match(line)
        if m:
            rows.append(tuple(float(v) for v in m.groups()))
    segments = [(b[0]-a[0], b[1]-a[1], b[2]-a[2]) for a, b in zip(rows, rows[1:])
                if b[2] > a[2] and b[0] > a[0]]
    if not segments:
        return {}
    values = np.array(segments)
    speed = values[:, 0]/values[:, 2]
    return dict(logged_iterations_per_second=float(values[:, 0].sum()/values[:, 2].sum()),
                objective_gain_per_second=float(values[:, 1].sum()/values[:, 2].sum()),
                rate_min=float(speed.min()), rate_p50=float(np.median(speed)),
                rate_p95=float(np.quantile(speed, .95)), rate_max=float(speed.max()),
                logged_intervals=len(segments))


def stop_previous(pid, expected_command):
    """Only terminate the explicitly identified previous workflow, never a pattern."""
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists():
        return False
    command = path.read_bytes().replace(b'\0', b' ').decode().strip()
    if command != expected_command or pid == os.getpid():
        raise ValueError(f'Refusing to stop unexpected process {pid}: {command}')
    os.kill(pid, signal.SIGTERM)
    return True


def benchmark(node_count, g, root, counts, iterations, *, continue_solve=True,
              stop_pid=None, expected_command=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if not counts or any(t < 1 for t in counts) or iterations < 1:
        raise ValueError('Positive thread counts and iteration budget required')
    started = time.monotonic()
    core._Highs.resetGlobalScheduler(True)
    original = core._Highs()
    options(original, 1, root/'presolve.log', presolve=True)
    checked(original.passModel(model(node_count, g)))
    write(root/'progress.json', dict(stage='presolve'))
    checked(original.presolve())
    status = original.getModelPresolveStatus()
    if status not in (core.HighsPresolveStatus.kReduced, core.HighsPresolveStatus.kNotReduced):
        raise RuntimeError(f'Unexpected presolve status: {status}')
    reduced = original.getPresolvedLp()
    write(root/'presolve.json', dict(seconds=time.monotonic()-started,
          rows=reduced.num_row_, columns=reduced.num_col_, highs=original.version()))
    # Leave the earlier solve running until all expensive replacement preparation
    # has succeeded. Then remove CPU/memory-bandwidth interference for benchmarks.
    if stop_pid is not None:
        write(root/'previous_process.json', dict(pid=stop_pid,
              stopped=stop_previous(stop_pid, expected_command)))
    trials = []; best_basis = None; best_score = -np.inf; best = None
    for threads in counts:
        core._Highs.resetGlobalScheduler(True)
        h = core._Highs()
        log = root/f'threads_{threads}.log'
        options(h, threads, log, limit=iterations)
        checked(h.passModel(reduced))
        write(root/'progress.json', dict(stage='benchmark', threads=threads, completed=trials))
        begin = time.monotonic(); checked(h.run()); seconds = time.monotonic()-begin
        info = h.getInfo(); status = h.getModelStatus()
        if status not in (core.HighsModelStatus.kOptimal, core.HighsModelStatus.kIterationLimit):
            raise RuntimeError(f'Thread trial {threads}: {h.modelStatusToString(status)}')
        trial = dict(threads=threads, simplex_concurrency=min(threads, 8), seconds=seconds, iterations=info.simplex_iteration_count,
                     iterations_per_second=info.simplex_iteration_count/seconds,
                     objective=info.objective_function_value,
                     infeasible_entries=info.num_primal_infeasibilities,
                     infeasibility_sum=info.sum_primal_infeasibilities,
                     optimal=status == core.HighsModelStatus.kOptimal, **log_rates(log))
        trials.append(trial)
        # At equal pivot budgets select actual objective progress throughput, not
        # cheaper but unproductive pivots. Tiny completed regression models use time.
        score = 1/seconds if trial['optimal'] else trial.get('objective_gain_per_second', 0.)
        if best is None or score > best_score:
            best = trial; best_score = score; best_basis = h.getBasis()
            checked(h.writeBasis(str(root/'best.basis')))
        write(root/'trials.json', dict(trials=trials, selected=best))
        print(json.dumps(trial), flush=True)
        checked(h.clear()); del h
    write(root/'selected.json', best)
    if not continue_solve:
        return dict(trials=trials, selected=best)
    core._Highs.resetGlobalScheduler(True)
    h = core._Highs(); options(h, best['threads'], root/'solver.log')
    checked(h.passModel(reduced)); checked(h.setBasis(best_basis))
    write(root/'progress.json', dict(stage='solve', selected=best, trials=trials))
    checked(h.run())
    if h.getModelStatus() != core.HighsModelStatus.kOptimal:
        raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
    checked(original.setOptionValue('threads', best['threads']))
    checked(original.postsolve(h.getSolution(), h.getBasis()))
    solution = original.getSolution()
    if not solution.value_valid or not solution.dual_valid:
        raise RuntimeError('Postsolve did not recover primal and dual values')
    result = certify_flow(node_count, g['a'], g['b'], g['target'], g['weight'],
        np.zeros(len(g['a']), bool), -np.asarray(solution.row_dual), np.asarray(solution.col_value),
        seconds=time.monotonic()-started,
        iterations=best['iterations']+h.getInfo().simplex_iteration_count)
    result['certificate']['backend'] = 'SciPy bundled HiGHS simplex thread selection'
    return dict(trials=trials, selected=best, result=result)


def run(config):
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    source = Path(config['source_root'])
    signature = dict(config=config, graph=dict(size=(source/'graph.npz').stat().st_size,
                     mtime_ns=(source/'graph.npz').stat().st_mtime_ns))
    if (root/'request.json').exists():
        raise ValueError('Use a new output root; benchmark is not resumable')
    write(root/'request.json', signature)
    if config.get('experiment_link'):
        Path(config['experiment_link']).symlink_to(root, target_is_directory=True)
    catalog = arrays(source/'geometry/catalog.npz')
    g = arrays(source/'graph.npz')
    outcome = benchmark(len(catalog['source_id']), g, root, config.get('thread_counts', [1,2,4,8]),
                        config.get('iterations', 60000), stop_pid=config.get('stop_previous_pid'),
                        expected_command=config.get('previous_command'))
    result = outcome.pop('result')
    np.savez(root/'solution.npz', source_id=catalog['source_id'], family=catalog['family'],
             **{k:v for k,v in result.items() if k!='certificate'})
    summary = dict(fibers=len(catalog['source_id']), arcs=len(g['a']),
                   certificate=result['certificate'], **outcome)
    write(root/'summary.json', summary)
    write(root/'progress.json', dict(stage='complete', **summary))
    return summary
