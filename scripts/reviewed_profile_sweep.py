"""Full reviewed-cohort output-chunk sweep using the shared evaluation runner."""
import copy
import json
import subprocess
import time
from pathlib import Path

from reviewed_fiber_evaluation import ROOT, execute_evaluation
from lasagna.inference_provenance import atomic_write


def variant(base, chunk, root):
    config = copy.deepcopy(base)
    config['output_root'] = str(root / f'chunk{chunk}')
    evaluation = config['evaluation']
    evaluation['inference_cache_namespace'] = f'reviewed_u31500_profile_v1_c{chunk}'
    evaluation['gpu_manager']['label'] = f'reviewed-profile-c{chunk}'
    settings = evaluation['inference']
    for key in ('accumulator_hot_slots', 'accumulator_spill_segment_slots'):
        settings[key] = settings[key] * 64**3 // chunk**3
    settings.update(chunk_size=chunk, profile_pipeline=True)
    return config


def check_fresh_or_complete(output):
    full = output / 'full'
    if not full.exists():
        return False
    summary = full / 'summary.json'
    if not summary.exists() or not json.loads(summary.read_text()).get('completed'):
        raise RuntimeError('Partial profiling run: use a new namespace/version; resuming would warm the inference cache')
    for name in ('pipeline_profile.json', 'inference_summary.json', 'inference_queues.jsonl'):
        if not (full/name).is_file():
            raise RuntimeError(f'Completed run missing required profile artifact: {name}')
    if not (output/'gpu_utilization.csv').is_file():
        raise RuntimeError('Completed run missing GPU telemetry')
    return True


def main():
    root = ROOT / 'profile_chunks_u31500_v1'
    base = json.loads((ROOT / 'halfsigma_current_raw_v3/workload.json').read_text())
    for chunk in (64, 32, 128):
        config = variant(base, chunk, root)
        output = Path(config['output_root'])
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'workload.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('profile configuration changed: use new namespace')
        atomic_write(path, config, durable=False)
        if check_fresh_or_complete(output):
            execute_evaluation(path, config, f'profile-c{chunk}')
            continue
        start = time.time()
        with (output / 'gpu_utilization.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '-i', '0,1,2,3,4,5',
                '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
                '--format=csv,nounits', '-l', '1'], stdout=log, stderr=subprocess.STDOUT)
            try:
                execute_evaluation(path, config, f'profile-c{chunk}')
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
                atomic_write(output / 'monitor_window.json',
                    {'start_unix': start, 'end_unix': time.time(), 'monitor_returncode': monitor.returncode}, durable=False)


if __name__ == '__main__':
    main()
