"""Approved-runner driver for the two supplied neural followers."""
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'vesuvius/src'),str(REPO/'gpu_manager/src'),str(REPO)]
MODEL_ROOT=Path('/ephemeral/hendrik/fiber3d_train2/models/neural_follow_20261006')
EVAL_ROOT=Path('/ephemeral/hendrik/fiber3d_train2/evaluations/neural_follow_91_20261007_v1')


def main(control):
    if control.get('operation') == 'report':
        from neural_follow_report import report
        report(EVAL_ROOT)
        return
    if control.get('operation') == 'build':
        subprocess.run(['bash',str(REPO/'vesuvius/scripts/run_fiber3d_native_build')],check=True)
        subprocess.run(['cmake','--build',str(REPO/'volume-cartographer/build/fiber-evaluation'), '--target','vc_volume','-j','8'],check=True)
        return
    if control.get('operation') == 'download':
        import boto3
        from lasagna.inference_provenance import atomic_write, sha256_file
        client = boto3.Session(profile_name='default').client('s3')
        records=[]
        for directory in ('flow_v5_100626', 'unified_scoring_v1_100726', 'heading_model_100526'):
            prefix='datasets/models/fiber_tracing/'+directory+'/'
            for page in client.get_paginator('list_objects_v2').paginate(Bucket='philodemos', Prefix=prefix):
                for obj in page.get('Contents', []):
                    relative=Path(obj['Key']).relative_to(prefix)
                    if relative.suffix not in ('.pt','.ckpt','.json'): continue
                    target=MODEL_ROOT/directory/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        part=target.with_suffix(target.suffix+'.partial')
                        client.download_file('philodemos',obj['Key'],str(part))
                        part.rename(target)
                    if target.stat().st_size!=obj['Size']: raise ValueError('checkpoint size mismatch')
                    row=dict(key=obj['Key'],path=str(target),bytes=obj['Size'],sha256=sha256_file(target))
                    records.append(row)
                    print(json.dumps(row),flush=True)
        atomic_write(MODEL_ROOT/'new_models_manifest.json',records,durable=False)
        return
    from lasagna.inference_provenance import atomic_write
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['FIBER_FOLLOW_NATIVE_DIR']=str(REPO/'volume-cartographer/build/fiber-evaluation/python/vc')
    env['PYTHONPATH']=os.pathsep.join([str(REPO/'volume-cartographer/build/fiber-evaluation/python'),
        str(REPO/'vesuvius/src'),str(REPO/'gpu_manager/src'),str(REPO),str(REPO/'scripts')])
    if control.get('tests'):
        upstream=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_follow/tests'
        subprocess.run([sys.executable,'-m','pytest','--noconftest','-q','-o','pythonpath=',
            'vesuvius/tests/neural_tracing/test_neural_follow_evaluate.py',
            'vesuvius/tests/neural_tracing/test_short_output_segments.py',
            *[str(upstream/name) for name in ('test_policy.py','test_flow.py','test_trace_sampling.py')]],env=env,cwd=REPO,check=True)
        return
    models={'flow_v5':('flow_v5_100626/flow_v5_scratch_huber_d144_ckpt_026000_s3.pt',dict(confidence=.5,n_commit=8,gate='full')),
            'unified':('unified_scoring_v1_100726/ckpt_035000.pt',dict(confidence=.7,n_commit=8,gate='full'))}
    for name in control.get('models',['flow_v5','unified']):
        checkpoint,overrides=models[name]
        output=EVAL_ROOT/(name+('-pilot' if control.get('pilot',True) else '-full'))
        config=dict(checkpoint=str(MODEL_ROOT/checkpoint),heading_checkpoint=str(MODEL_ROOT/'heading_model_100526/ckpt_064000.pt'),
            output=str(output),policy_overrides=overrides,precision='bf16',
            reference_config='/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reviewed_ema999_trace_tuning_20261006_v1/p40_cos40_norev_step6/full.json',
            expected_cases=182,native_binary=str(REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'),
            trace_grid_base=8,ct_grid_base=4,ct_level=2,max_batch_size=32 if name=='unified' else 16,
            backend='legacy' if name=='flow_v5' else 'current',sampling_seed=0,max_len=100000,
            case_workers_per_gpu=32,input_workers_per_gpu=24,io_threads_per_gpu=24,
            replay=dict(nominal_step_base=8,threshold_base=40,grace_base=160,tangential_factor=2,match_refine_steps=1),
            gpu_manager=dict(enabled=True,exclusive=True,priority=100,cpu_cores=192,label=f'neural-follow-{name}'))
        if control.get('pilot',True): config['case_ids']=control.get('case_ids',[0,1])
        path=output/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError('neural evaluation config changed; use new namespace')
        atomic_write(path,config,durable=False)
        subprocess.run([sys.executable,'-u','-m','vesuvius.neural_tracing.fiber_trace_3d.neural_follow_evaluate',
            '--config',str(path),'--devices',control.get('devices','0,1,2,3,4,5')],env=env,cwd=REPO,check=True)
        if not control.get('pilot',True):
            from neural_follow_report import report
            report(EVAL_ROOT)


if __name__=='__main__': main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
