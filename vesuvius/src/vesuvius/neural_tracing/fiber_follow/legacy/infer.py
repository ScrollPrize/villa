"""Trace with a pre-cleanup checkpoint ('flow_matching': legacy/flow_v5.py, 'coordinate_regression':
legacy/coordinate_regression.py, chosen by the checkpoint's model type); arguments as tracing/infer.py."""
import torch

from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
from vesuvius.neural_tracing.fiber_follow.legacy import coordinate_regression, flow_v5
from vesuvius.neural_tracing.fiber_follow.tracing.inference import main as infer

LOADERS = {'flow_matching': flow_v5.load_checkpoint, 'coordinate_regression': coordinate_regression.load_checkpoint}


def load_checkpoint(path, device='cuda'):
    model_type = torch.load(path, map_location='cpu', weights_only=False, mmap=True).get('model_type')
    if model_type not in LOADERS:
        raise ValueError(f'{path}: model type {model_type!r} is not a supported pre-cleanup checkpoint ({", ".join(LOADERS)})')
    return LOADERS[model_type](path, device)


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=FiberTracer)


if __name__ == '__main__':
    main()
