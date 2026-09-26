"""Use the frozen calibration/final protocol for the direct follower."""
from evaluate_single_path import main
from vesuvius.neural_tracing.fiber_follow.direct.train import load_checkpoint, validate_volume_source
from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer

if __name__ == '__main__':
    import sys
    import json
    from pathlib import Path
    judge_mode = '--judge' in sys.argv
    if '--selection' in sys.argv:
        selection = json.loads(Path(sys.argv[sys.argv.index('--selection')+1]).read_text())
        judge_mode |= 'judge_policy' in selection
    if judge_mode:
        from vesuvius.neural_tracing.fiber_follow.direct.judge_calibrate import main as judge_main
        judge_main()
    else:
        main(checkpoint_loader=load_checkpoint, model_tracer=DirectTracer, volume_validator=validate_volume_source)
