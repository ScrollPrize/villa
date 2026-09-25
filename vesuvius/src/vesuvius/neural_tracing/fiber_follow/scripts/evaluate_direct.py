"""Use the frozen calibration/final protocol for the direct follower."""
from evaluate_single_path import main
from vesuvius.neural_tracing.fiber_follow.direct.train import load_checkpoint, validate_volume_source
from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer

if __name__ == '__main__':
    main(checkpoint_loader=load_checkpoint, model_tracer=DirectTracer, volume_validator=validate_volume_source)
