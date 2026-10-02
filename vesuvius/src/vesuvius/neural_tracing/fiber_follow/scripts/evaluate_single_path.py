"""The flow-matching follower's evaluation, with the shared protocol on frozen Paris 4 seeds.

  python scripts/evaluate_single_path.py calibrate --checkpoint RUN/ckpt.pt --out EVAL
  python scripts/evaluate_single_path.py run --checkpoint RUN/ckpt.pt --policy EVAL/selection.json --out EVAL/final.json
"""
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.flow_matching.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand, fiber_manifest, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.evaluation import main as protocol
from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec

FIBERS = '/mnt/raid_nvme/spiral_dataset_working/fibers'
MANIFEST = Path(__file__).parents[1]/'output'/'single_path_v11_preparation'/'seeds.json'


def paris_sources(args):
    _, _, _, spec, _ = load_checkpoint(args.checkpoint, 'cpu')
    manifest = read_manifest(MANIFEST)
    if spec != FiberVolumeSpec(**manifest['volume']):
        raise ValueError('Volume differs from frozen manifest')
    _, validation = split_fibers(load_fibers(FIBERS, grid_scale=spec.grid_scale),
                                 ZBand(45000/spec.grid_scale, 48500/spec.grid_scale))
    if fiber_manifest(validation) != manifest['fibers']:
        raise ValueError('Geometry differs from frozen manifest')
    return [dict(name='paris4', fibers=validation, manifest=manifest, detector=None, volume=FiberVolume(spec))]


if __name__ == '__main__':
    protocol(checkpoint_loader=load_checkpoint, tracer_class=ModelTracer, source_loader=paris_sources)
