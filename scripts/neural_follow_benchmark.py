"""Approved-runner driver for supplied neural followers."""
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'vesuvius/src'),str(REPO/'gpu_manager/src'),str(REPO)]
MODEL_ROOT=Path('/ephemeral/hendrik/fiber3d_train2/models/neural_follow_20261006')
EVAL_ROOT=Path('/ephemeral/hendrik/fiber3d_train2/evaluations/neural_follow_91_20261007_v1')
HISTORICAL_ROOT=EVAL_ROOT.with_name('neural_follow_91_20261006_v2')
HISTORICAL_COMMIT='ba86eea12231ff770324d2adb7fceb709e93a3ea'
HISTORICAL_RUNTIME=MODEL_ROOT/('runtime_'+HISTORICAL_COMMIT[:9])


def historical_config(name, output):
    """Preserve the original checkpoint, geometry, policy and batching."""
    config=json.loads((HISTORICAL_ROOT/(name+'-full')/'config.json').read_text())
    config.update(output=str(output),backend='historical',sampling_seed=0,
        forward_length_mode='reference_remaining_x3',upstream='9200d51a9',
        follower_runtime=str(HISTORICAL_RUNTIME),follower_runtime_commit=HISTORICAL_COMMIT)
    config['native_binary']=str(REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric')
    return config


def main(control):
    if control.get('older_matrix'):
        for name in ('memory','flow'):
            for reverse in (False,True):
                main({**control,'older_matrix':False,'models':[name],'pilot':False,'reverse':reverse})
        return
    eval_root = EVAL_ROOT if not control.get('reverse') else EVAL_ROOT.with_name('neural_follow_reverse_91_20261007_v1')
    if control.get('operation') == 'inspect_older':
        import io
        import tarfile
        from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_evaluate import activate_follower_runtime,digest
        archive=subprocess.check_output(['git','archive',HISTORICAL_COMMIT,
            'vesuvius/src/vesuvius/neural_tracing/fiber_follow'],cwd=REPO)
        HISTORICAL_RUNTIME.mkdir(parents=True,exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(HISTORICAL_RUNTIME,filter='data')
        activate_follower_runtime(historical_config('memory','/unused'))
        import torch
        torch.set_num_threads(1)
        from vesuvius.neural_tracing.fiber_follow.train.train import load_checkpoint
        from vesuvius.neural_tracing.fiber_follow.tracing.policy import checkpoint_policy
        for name in ('memory','flow'):
            cfg=historical_config(name,'/unused')
            old=json.loads((HISTORICAL_ROOT/(name+'-full')/'summary.json').read_text())['neural_follow']
            compared=0
            for path,expected in old['implementation'].items():
                marker='vesuvius/src/vesuvius/neural_tracing/fiber_follow/'
                if marker not in path: continue
                actual=HISTORICAL_RUNTIME/marker/path.split(marker,1)[1]
                if digest(actual)!=expected: raise ValueError('historical runtime mismatch: '+str(actual))
                compared+=1
            if not compared: raise ValueError('no historical implementation hashes checked')
            model,crop,history,volume,checkpoint=load_checkpoint(cfg['checkpoint'],'cpu')
            print(name,json.dumps(dict(model_type=checkpoint.get('model_type'),
                heading_sha256=model.cfg.frame_checkpoint_sha256,
                policy=vars(checkpoint_policy(checkpoint,model.cfg,**cfg['policy_overrides'])),
                history=history,verified_runtime_files=compared),default=str),flush=True)
        return
    if control.get('operation') == 'report':
        from neural_follow_report import report
        report(eval_root,reverse=bool(control.get('reverse')))
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
        for name in ('test_fiber_trace3d','test_fiber_replay','test_fiber_reference_replay_benchmark'):
            subprocess.run([str(REPO/'volume-cartographer/build/fiber-evaluation/bin'/name)],env=env,cwd=REPO,check=True)
        upstream=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_follow/tests'
        subprocess.run([sys.executable,'-m','pytest','--noconftest','-q','-o','pythonpath=',
            'vesuvius/tests/neural_tracing/test_neural_follow_evaluate.py',
            'vesuvius/tests/neural_tracing/test_short_output_segments.py',
            *[str(upstream/name) for name in ('test_policy.py','test_flow.py','test_trace_sampling.py')]],env=env,cwd=REPO,check=True)
        return
    models={'flow_v5':('flow_v5_100626/flow_v5_scratch_huber_d144_ckpt_026000_s3.pt',dict(confidence=.5,n_commit=8,gate='full')),
            'unified':('unified_scoring_v1_100726/ckpt_035000.pt',dict(confidence=.7,n_commit=8,gate='full'))}
    for name in control.get('models',['flow_v5','unified']):
        older=name in ('memory','flow')
        if not older and name not in models: raise ValueError('unknown neural model: '+name)
        checkpoint,overrides=models.get(name,('',{}))
        # Keep old historical outputs immutable; fresh controls share current code.
        target_root=EVAL_ROOT.with_name('neural_follow_reverse_91_20261007_v1') if older else eval_root
        suffix='-control' if older and not control.get('reverse') else ''
        output=target_root/(name+suffix+('-pilot' if control.get('pilot',True) else '-full'))
        config=dict(checkpoint=str(MODEL_ROOT/checkpoint),heading_checkpoint=str(MODEL_ROOT/'heading_model_100526/ckpt_064000.pt'),
            output=str(output),policy_overrides=overrides,precision='bf16',
            reference_config='/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reviewed_ema999_trace_tuning_20261006_v1/p40_cos40_norev_step6/full.json',
            expected_cases=182,native_binary=str(REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'),
            trace_grid_base=8,ct_grid_base=4,ct_level=2,max_batch_size=32 if name=='unified' else 16,
            backend='legacy' if name=='flow_v5' else 'current',sampling_seed=0,max_len=100000,
            case_workers_per_gpu=32,input_workers_per_gpu=24,io_threads_per_gpu=24,
            replay=dict(nominal_step_base=8,threshold_base=40,grace_base=160,tangential_factor=2,match_refine_steps=1),
            gpu_manager=dict(enabled=True,exclusive=True,priority=100,cpu_cores=192,label=f'neural-follow-{name}'))
        if older: config=historical_config(name,output)
        if control.get('pilot',True): config['case_ids']=control.get('case_ids',[0,1])
        if control.get('reverse'):
            config['replay'].update(reverse_interval_base=512,reverse_normal_radius_base=20,
                reverse_tangent_radius_base=40,reverse_to_start=True,reverse_all_endpoints=True,
                reverse_history_steps=0,reseed='reference_arc_without_dense_presence')
        path=output/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError('neural evaluation config changed; use new namespace')
        atomic_write(path,config,durable=False)
        subprocess.run([sys.executable,'-u','-m','vesuvius.neural_tracing.fiber_trace_3d.neural_follow_evaluate',
            '--config',str(path),'--devices',control.get('devices','0,1,2,3,4,5')],env=env,cwd=REPO,check=True)
        if not control.get('pilot',True):
            from neural_follow_report import report
            report(target_root,reverse=bool(control.get('reverse')) or older)


if __name__=='__main__': main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
