"""Fresh interleaved fp16/fp32 repeatability trials, shared evaluation executor."""
import copy
import json
import subprocess
from pathlib import Path
from reviewed_fiber_evaluation import ROOT, execute_evaluation
from lasagna.inference_provenance import atomic_write

EXPERIMENT = ROOT/'accumulation_repeats_u31500_v1'


def check_complete(output):
    full=output/'full'
    if not full.exists():
        return False
    for name in ('summary.json','inference_summary.json','pipeline_profile.json'):
        if not (full/name).is_file():
            raise RuntimeError(f'partial run missing {name}; do not reuse warm inference')
    if not json.loads((full/'summary.json').read_text()).get('completed'):
        raise RuntimeError('incomplete run; do not reuse warm inference')
    return True


def variant(base, dtype, repeat):
    if dtype not in ('float16','float32') or repeat not in (1,2,3):
        raise ValueError('unsupported repeat')
    config = copy.deepcopy(base)
    name = f'{dtype}_r{repeat}'
    config['output_root'] = str(EXPERIMENT/name)
    evaluation = config['evaluation']
    evaluation['inference_cache_namespace'] = f'reviewed_u31500_accum_v1_{name}'
    evaluation['gpu_manager']['label'] = f'reviewed-accum-{name}'
    settings = evaluation['inference']
    assert settings['chunk_size'] == 64 and settings['accumulator_dtype'] == 'float16'
    assert not settings['emit_embeddings']
    # Frozen model: seven product planes + one weight plane (always fp32).
    # Record payload 18 -> 32 bytes/voxel, not a factor of two.
    if dtype == 'float32':
        for key in ('accumulator_hot_slots','accumulator_spill_segment_slots'):
            assert settings[key]*9 % 16 == 0
            settings[key] = settings[key]*9//16
    settings.update(accumulator_dtype=dtype, profile_pipeline=True)
    return config


def main():
    base = json.loads((ROOT/'halfsigma_current_raw_v3/workload.json').read_text())
    for repeat in (1,2,3):
        for dtype in ('float16','float32'):
            config = variant(base,dtype,repeat)
            output = Path(config['output_root'])
            output.mkdir(parents=True,exist_ok=True)
            path = output/'workload.json'
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError('immutable experiment config changed')
            if not check_complete(output):
                resolver=Path(config['native_binary']).with_name('vc_cache_root')
                cache_root=Path(subprocess.check_output([str(resolver)],text=True).strip())
                namespace=cache_root/'live_inference'/'_namespaces'/config['evaluation']['inference_cache_namespace']
                if namespace.exists():
                    raise RuntimeError('existing inference namespace without complete output; use new experiment version')
            atomic_write(path,config,durable=False)
            execute_evaluation(path,config,f'{dtype}_r{repeat}')
    from report_accumulation_repeats import report
    atomic_write(EXPERIMENT/'comparison.json',report(EXPERIMENT),durable=False)


if __name__ == '__main__':
    main()
