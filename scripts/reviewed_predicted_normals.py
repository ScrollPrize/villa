"""Matched reviewed tracing comparison using the auxiliary normal head."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'vesuvius/src'), str(REPO/'gpu_manager/src'), str(REPO)]
from reviewed_194246_eval import BASE, NAME as PREVIOUS
from reviewed_trace_tuning import ROOT, DOCS, collect_filtered_result, document
from reviewed_reverse_prefix import evaluate_config
from lasagna.inference_provenance import atomic_write, sha256_file

NAME = 'reviewed_predicted_normals_20261007_v1'
CONTROLS = [('EMA999 u12100', BASE, 'Previous EMA999 u12100'),
            ('EMA99 u19100', ROOT/PREVIOUS/'best_loss_ema99_eval', 'best_loss_ema99 EMA99 u19100')]


def configuration(base, label):
    cfg = copy.deepcopy(base)
    cfg['output_root'] = str(ROOT/NAME/label)
    cfg['evaluation']['tracing_normals'] = 'fiber'
    cfg['evaluation']['inference']['emit_normals'] = True
    cfg['evaluation']['gpu_manager']['label'] = NAME+'_'+label
    return cfg


def main(control):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO/'vesuvius/src'),str(REPO),str(REPO/'scripts'),str(REPO/'gpu_manager/src')]),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if control.get('operation') == 'build':
        subprocess.run(['bash',str(REPO/'vesuvius/scripts/run_fiber3d_native_build')], check=True, env=env)
        return
    if control.get('operation') == 'tests':
        subprocess.run([sys.executable,'-m','pytest','--noconftest','-q','-o','pythonpath=',
            'vesuvius/tests/neural_tracing/test_predicted_normals.py',
            'vesuvius/tests/neural_tracing/test_width_tracer.py',
            'scripts/test_reviewed_194246_eval.py'],check=True,env=env)
        subprocess.run([str(REPO/'volume-cartographer/build/fiber-evaluation/bin/test_fiber_trace3d')],check=True,env=env)
        return
    saved = json.loads((DOCS/(PREVIOUS+'.json')).read_text())
    paired = bool(control.get('paired_controls'))
    report_path = DOCS/(NAME+'.json')
    rows = json.loads(report_path.read_text()) if report_path.exists() else {}
    for index, (label, folder, key) in enumerate(CONTROLS):
        base = json.loads((folder/'workload.json').read_text())
        if sha256_file(base['checkpoint']) != base['checkpoint_sha256']:
            raise ValueError('frozen checkpoint changed')
        rows[label+' Lasagna'] = saved[key]
        cfg = configuration(base, str(index)+('_lasagna' if paired else ''))
        if paired:
            cfg['evaluation']['tracing_normals'] = 'lasagna'
        output = Path(cfg['output_root'])
        summary = output/'full/summary.json'
        if (output/'full').exists() and not (summary.exists() and json.loads(summary.read_text()).get('completed')):
            from training_failure_mining import archive_incomplete_output
            print('Archived incomplete attempt: '+str(archive_incomplete_output(output/'full')),flush=True)
        evaluate_config(cfg, NAME+'_'+str(index))
        output = Path(cfg['output_root'])
        effective = json.loads((output/'full.json').read_text())
        previous = json.loads((folder/'full.json').read_text())
        def cohort(c):
            return [(s['id'],s['fibers'],s['normal_manifest'],s['coordinate_space']) for s in c['sources']]
        if effective['trace_args'] != previous['trace_args'] or cohort(effective) != cohort(previous):
            raise ValueError('control policy or cohort changed')
        old_binding = json.loads((folder/'full_bindings.json').read_text())
        new_binding = json.loads((output/'full_bindings.json').read_text())
        if old_binding['sources'] != new_binding['sources']:
            raise ValueError('reference geometry/content or normal binding changed')
        result, metrics, cases = collect_filtered_result(output)
        native = json.loads((output/'full/cases/0/trace.json').read_text())
        if paired:
            original = json.loads((ROOT/NAME/str(index)/'full/summary.json').read_text())
            matched = json.loads((output/'full/summary.json').read_text())
            def manifests(summary):
                return {s['id']:s['fiber_manifest'] for s in summary['config']['sources']}
            if manifests(original) != manifests(matched):
                raise ValueError('paired control did not reuse identical prediction caches')
        mode = 'lasagna' if paired else 'fiber'
        if native['provenance'].get('tracing_normals', 'lasagna') != mode:
            raise ValueError('requested normal mode was not applied')
        if native['provenance']['effective_trace'] != saved[key]['effective_trace']:
            raise ValueError('effective native tuning changed')
        rows[label+(' Lasagna matched cache' if paired else ' fiber normals')] = dict(metrics=metrics,changes={'tracing_normals':mode},
            summary_path=result['summary_path'],checkpoint_sha256=base['checkpoint_sha256'],
            effective_trace=native['provenance']['effective_trace'], trace_args=effective['trace_args'],
            wall_seconds=json.loads((output/'timing.json').read_text())['wall_seconds'],
            native_sha256_at_analysis=sha256_file(cfg['native_binary']))
        atomic_write(output/'piece_decisions.json',cases,durable=False)
        document(rows,name=NAME,title='Predicted versus Lasagna tracing normals',action='reviewed_predicted_normals',
            notes=['Only tracing normal source changes. GT acceptance and restart search keep Lasagna normals. Same frozen checkpoints, widths, cutoff and step6; FW/BW disabled in both winning controls.',
                   'Fresh auxiliary-normal caches required: elapsed times include inference and are not isolated tracing speed comparisons. Historical controls retained; matched-cache Lasagna controls reuse identical presence/direction/width bytes. GPU0–5, unchanged worker/batch settings.'])
        print('COMPLETED '+label+' '+mode+' '+json.dumps(metrics['512']['combined']),flush=True)


if __name__ == '__main__':
    main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
