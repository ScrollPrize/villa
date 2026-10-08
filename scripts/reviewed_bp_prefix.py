"""Frozen raw40500, matched1024 BP crop trial through existing fast machinery."""
import itertools
import json
import math
import re
import statistics
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'vesuvius/src'), str(REPO), str(REPO/'gpu_manager/src')]
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import CachedReplaySession
from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import demand_gpu_manager_settings
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run, bp_metrics

ROOT = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_sideband_best_raw_20261002')
OUT = Path('/ephemeral/hendrik/fiber_crop_bench/20261002_raw40500_prefix512_v1')
CONFIGS = REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs'
LO, HI = (10240,22016,6144), (11264,23040,7168)


def save_config(path, config):
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f'configuration changed: use a new output namespace, not stale outputs: {path}')
    atomic_write(path, config, durable=False)


def report():
    rows = []
    historical = Path('/ephemeral/hendrik/fiber_crop_bench/20260923_v1/bp-beam-new-perpsign-p1hardsign.log')
    if historical.exists():
        rows.append(dict(policy='Historical u22000, hard-sign BP (different trace coverage)', **bp_metrics(historical.read_text())))
    for policy in ('unfiltered', 'prefix512'):
        attrs = json.loads((OUT/f'beam-{policy}.zarr/.zattrs').read_text())
        artifact = attrs['processing']['artifact']
        summary = artifact['trace_summary']
        retained, skipped = summary['retained_length_base'], summary['reverse_skipped_length_base']
        rows.append(dict(policy=policy, **bp_metrics((OUT/f'bp-{policy}.log').read_text()),
            trace_count=artifact['trace_count'],
            retained_length_base=retained, discarded_length_base=skipped,
            retained_of_own_candidates_percent=100*retained/(retained+skipped) if retained+skipped else 0,
            reverse_attempts=summary['reverse_attempts'], reverse_failures=summary['reverse_failures'],
            candidate_batch_seconds=summary['candidate_batch_seconds'],
            tracing_seconds=json.loads((OUT/f'beam-{policy}.run.json').read_text())['wall_seconds'],
            bp_seconds=json.loads((OUT/f'bp-{policy}.run.json').read_text())['wall_seconds']))
    owner_times = [float(v) for v in re.findall(r'stage=anchors .*? seconds=([\d.]+)', (OUT/'beam-unfiltered.log').read_text())]
    perf = dict(owner_chunks=len(owner_times))
    if owner_times:
        import numpy as np
        perf.update(mean_seconds=statistics.mean(owner_times), p50_seconds=float(np.percentile(owner_times,50)),
                    p95_seconds=float(np.percentile(owner_times,95)))
    data = dict(rows=rows, anchor_preparation=perf, output=str(OUT))
    atomic_write(OUT/'comparison.json',data,durable=False)
    atomic_write(REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-prefix512.json',data,durable=False)
    table = ['\n## Completed results\n',
        '| Variant | Constraint score % (closer to0 better) | Oracle-reference exact % | Traces | Own-candidate retained % | Trace / BP seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['policy']} | {row['constraint_score_percent']:.4f}% | {row['reference_score_percent']:.4f}% | " +
            (f"{row['trace_count']} | {row['retained_of_own_candidates_percent']:.4f}% | {row['tracing_seconds']:.2f} / {row['bp_seconds']:.2f} |"
             if 'trace_count' in row else '— | — | — |'))
    table += ['', 'Retained percentage uses each variant’s own integrated candidates, not GT length or the other variant’s paths. Different suppression decisions can change total traced length. Exact numerators/denominators and base lengths are in the adjacent JSON.',
              '', f'Anchor preparation,128configured workers: {json.dumps(perf)}. Cold serial attempt was interrupted at164s before its first owner completed; this is a lower-bound comparison, not a completed serial throughput measurement. Filtered timing reuses these anchors. One repetition per variant, RelWithDebInfo.',
              '', 'Verification:23replay,90trace,26reference,86crop,86anchor native test cases and4Python queue tests passed. Anchor fixture compares1and128configured workers; no completed real-data serial-owner hash comparison was performed.']
    path = REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-prefix512.md'
    text = path.read_text().split('\n## Completed results')[0].replace('Status: running. Results will be appended after completion.', 'Status: completed.')
    path.write_text(text+'\n'.join(table)+'\n')
    print(json.dumps(data,indent=2),flush=True)


def prepare_coverage(lo=LO, hi=HI, out=OUT, root=ROOT, devices='0,1,2,3,4,5'):
    """Fill cached model predictions for a crop using the existing live pipeline."""
    out.mkdir(parents=True, exist_ok=True)
    workload = json.loads((root/'workload.json').read_text())
    if sha256_file(workload['checkpoint']) != workload['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    evaluation = json.loads((root/'full/summary.json').read_text())['config']
    evaluation['sources'] = [s for s in evaluation['sources'] if s['id'] == 'PHercParis4']
    if len(evaluation['sources']) != 1:
        raise ValueError('expected one canonical Paris4 source')
    source = evaluation['sources'][0]
    manifest = Path(source['fiber_manifest'])
    cache = manifest.parent
    identity = json.loads((cache/'cache_identity.json').read_text())
    factor = (1 << identity['identity']['source']['level']) * identity['geometry']['scale']
    chunk = identity['geometry']['chunk']
    # Conservative coverage for canonical anchor owner chunks and fit stencils,
    # plus free reverse beam lookahead. The dense paths themselves stop at crop.
    margin = 1024
    ranges = [range(max(0, math.floor((a-margin)/factor/chunk)),
                    math.ceil((b+margin)/factor/chunk)) for a,b in zip(lo,hi)]
    requests = list(itertools.product(*ranges[::-1]))
    manager = demand_gpu_manager_settings(workload['evaluation'], devices)
    with CachedReplaySession(evaluation, workload['native_binary'], workload['checkpoint'],
                             devices=devices, gpu_manager=manager) as session:
        filled = session.fill_chunks({str(cache): requests})
    coverage_path = out/('coverage_resume.json' if (out/'coverage.json').exists() else 'coverage.json')
    atomic_write(coverage_path, dict(**filled, margin_base=margin, factor=factor,
        chunk=chunk, minimum_base_xyz=lo, maximum_base_xyz=hi, checkpoint_sha256=workload['checkpoint_sha256']), durable=False)
    return workload, source, manifest, cache


def main():
    workload, source, manifest, cache = prepare_coverage()
    template = json.loads((CONFIGS/'crop_benchmark_beam_bp_p1hardsign_20260923.json').read_text())
    base = dict(output_root=str(OUT), reference_dir=template['reference_dir'],
                input_files=[str(manifest), source['normal_manifest']], measurement_date='2026-10-02')
    results = {}
    for policy in ('unfiltered', 'prefix512'):
        trace = OUT/f'beam-{policy}.zarr'
        name = f'beam-{policy}'
        args = ['trace','-', '--fiber-manifest',str(manifest),
                '--fiberlet-cache-root',str(cache/'bp_crop_anchors'),
                '--normal-manifest',source['normal_manifest'],
                '--bbox', *map(str,(*LO,*HI)), '--lookahead','384','--threads','128',
                '--output',str(trace),'--trace-tracer','beam',
                '--coverage','40','--coverage-tangential','80','--coverage-angle','45',
                '--inference-scaledown-power','0','--beam-width','8',
                '--beam-lookahead-steps','2','--step-voxels','4',
                '--smoothness-normal-weight','0.1','--smoothness-tangent-weight','10',
                '--embedding-weight','0']
        args += (['--no-reverse-consistency'] if policy == 'unfiltered' else
                 ['--reverse-consistency','--reverse-normal','20','--reverse-tangential','40',
                  '--reverse-prefix-interval','512'])
        command = dict(name=name, binary='vc_fiber_trace_chunk', args=args)
        config = dict(base, commands=[command])
        save_config(OUT/f'{name}.config.json', config)
        record = OUT/f'{name}.run.json'
        if not (record.exists() and json.loads(record.read_text())['returncode'] == 0 and trace.exists()):
            run(config)
        bpname = f'bp-{policy}'
        args = list(template['commands'][0]['args'])
        args[1] = str(trace)
        args[args.index('--output')+1] = str(OUT/bpname)
        config = dict(base, commands=[dict(name=bpname,binary='vc_fiber_trace_chunk',args=args)],
            publish_bp_plots=[dict(id=f'raw40500-1024-{policy}',label=f'Raw40500 beam8 {policy} (1024)',
                log=f'{bpname}.log',run_record='docs/fiber_benchmark_runs/2026-10-02-prefix512.md')])
        save_config(OUT/f'{bpname}.config.json',config)
        record = OUT/f'{bpname}.run.json'
        if record.exists() and json.loads(record.read_text())['returncode'] == 0:
            config['commands'] = []
        run(config)
        results[policy] = bp_metrics((OUT/f'{bpname}.log').read_text())
        atomic_write(OUT/'results.json',results,durable=False)
        print('BP RESULT '+policy+': '+json.dumps(results[policy]),flush=True)
    report()


if __name__ == '__main__':
    main()
