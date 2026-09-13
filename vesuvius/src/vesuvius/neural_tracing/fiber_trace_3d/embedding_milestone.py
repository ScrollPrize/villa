"""Wait without GPUs, freeze an exact checkpoint, then run existing managed analyses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def check_checkpoint(payload, step):
    actual = int(payload['step'])
    if actual > step:
        raise RuntimeError(f'Missed requested checkpoint {step}; current is {actual}')
    if actual < step:
        return False
    if payload.get('samples_seen') != step * 128 or payload.get('snapshot_kind', 'raw') != 'raw':
        raise ValueError('Checkpoint is not the requested raw 128-patch/update snapshot')
    return True


def run(control_path):
    import torch
    from .live_evaluate import freeze_checkpoint
    from villa_gpu_manager.processes import process_state
    torch.set_num_threads(1)
    control = json.loads(Path(control_path).read_text())
    settings = control['embedding_milestone']
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'status.json'
    def record(phase, **fields):
        value = dict(phase=phase, target_step=settings['step'], **fields)
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(status)
        print(json.dumps(value), flush=True)
    try:
        checkpoint = Path(settings['checkpoint'])
        identity = json.loads(Path(settings['training_identity']).read_text())
        previous = None
        while True:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            current = int(payload['step'])
            ready = check_checkpoint(payload, settings['step'])
            del payload
            if ready:
                frozen, digest = freeze_checkpoint(checkpoint, root / 'frozen')
                payload = torch.load(frozen, map_location='cpu', weights_only=True, mmap=True)
                if not check_checkpoint(payload, settings['step']):
                    raise RuntimeError('Frozen snapshot changed during capture')
                del payload
                record('checkpoint_captured', step=current, sha256=digest, checkpoint=str(frozen))
                break
            if process_state(identity, os.getuid()) != 'alive':
                raise RuntimeError('Training is no longer alive before the requested checkpoint')
            if current != previous:
                record('waiting', current_step=current)
                previous = current
            time.sleep(2)
        control['embedding_trace_pilot'].update(checkpoint=str(frozen), output=str(root / 'pilot'), resume=False)
        control['embedding_route_analysis'].update(pilot=str(root / 'pilot'), output=str(root / 'range'))
        frozen_control = root / 'analysis-control.json'
        frozen_control.write_text(json.dumps(control, indent=2))
        for module in ('embedding_trace_pilot', 'embedding_route_analysis'):
            record('analyzing', module=module, checkpoint=str(frozen), sha256=digest)
            subprocess.run([sys.executable, '-m', 'vesuvius.neural_tracing.fiber_trace_3d.' + module,
                            str(frozen_control)], check=True)
        new = json.loads((root / 'range/report.json').read_text())
        old = json.loads(Path(settings['baseline']).read_text())
        if old['step'] != new['step'] or old['fiber'] != new['fiber']:
            raise ValueError('Baseline checkpoint age/fiber mismatch')
        lines = ['# Matched-update embedding range comparison', '',
                 f'Both raw checkpoints: update {new["step"]}, {new["step"] * 128} patches.',
                 f'Old SHA: `{old["checkpoint_sha256"]}`. New SHA: `{digest}`.', '',
                 '| GT separation (base voxels) | Old mean cosine | Paired mean cosine |',
                 '|---:|---:|---:|']
        for lag in sorted(old['lags'], key=int):
            a, b = old['lags'][lag], new['lags'][lag]
            if a['requested'] != b['requested'] or a['n'] != b['n']:
                raise ValueError(f'GT coverage differs at lag {lag}; inspect raw results')
            lines.append(f'| {lag} | {a["mean"]:.4f} | {b["mean"]:.4f} |')
        lines += ['', 'One held-out fiber only. Failure-route events differ between models;',
                  'they are not a matched-event comparison. Full distributions and coverage are in range/report.json.']
        (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
        record('complete', report=str(root / 'comparison.md'), checkpoint=str(frozen), sha256=digest)
    except BaseException as exc:
        record('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control')
    run(parser.parse_args().control)
