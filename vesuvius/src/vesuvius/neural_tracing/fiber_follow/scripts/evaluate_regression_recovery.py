"""Evaluate direct checkpoints on frozen calibration recovery states."""
from evaluate_recovery import main
from vesuvius.neural_tracing.fiber_follow.train.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.observations import DirectTracer, ObservationBuilder

if __name__ == '__main__':
    main(checkpoint_loader=load_checkpoint, model_tracer=DirectTracer, batch_builder_factory=ObservationBuilder)
