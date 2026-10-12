"""What the tracing CLIs (inference.py, volume.py, service.py) share: the CT and frame-model options and their
resolution on this machine, and GPU crops.

A checkpoint records where it was trained: its CT (``vol_spec``: store, level, scales, chunk cache folder) and its
learned frame model (an absolute path pinned by sha256). On another machine:
- CT: ``--ct``/``--ct-level``/``--ct-grid-scale``/``--grid-scale`` override the recorded store and scales. A remote
  store (s3://, https://) needs a persistent chunk cache: ``--cache-dir``, else the recorded folder when it exists here,
  else ``$XDG_CACHE_HOME/vesuvius/volume_cache`` (``~/.cache/...``).
- Frame model: ``--frame-checkpoint``, else the recorded path (relocated by ``TRACERS_PATH_MAP``) when it exists, else
  a file next to the follower checkpoint: the recorded file name, then ``frame_model.pt``. A found file must have the
  recorded sha256.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FRAME_NAME = 'frame_model.pt'


def add_volume_arguments(ap):
    ap.add_argument('--ct', help="CT zarr (local path or s3://); default: the checkpoint's")
    ap.add_argument('--ct-level', type=int, help="CT pyramid level read; default: the checkpoint's")
    ap.add_argument('--ct-grid-scale', type=float, help='base voxels per CT voxel at that level (normally 2**level)')
    ap.add_argument('--grid-scale', type=float, help="base voxels per trace-grid voxel; default: the checkpoint's")
    ap.add_argument('--cache-dir', help='local cache for remote CT chunks (default: the recorded folder when it exists, '
                                        'else $XDG_CACHE_HOME/vesuvius/volume_cache)')
    ap.add_argument('--frame-checkpoint', help='learned frame model (default: the recorded path, else the recorded '
                                               f'file name or {DEFAULT_FRAME_NAME} next to --checkpoint)')


def default_cache_dir():
    return str(Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'vesuvius'/'volume_cache')


def resolve_frame_checkpoint(cfg, checkpoint, given=None):
    """Bind the follower's learned frame model on this machine (see the module notes); a no-op for CT frames."""
    from vesuvius.neural_tracing.tracers.fibers.frames.learned import bind_frame_checkpoint
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import file_sha256
    from vesuvius.neural_tracing.tracers.shared.paths import recorded_path
    if given:
        bind_frame_checkpoint(cfg, given)
        return
    recorded = getattr(cfg, 'frame_checkpoint', None)
    if not recorded:
        return
    if Path(recorded_path(recorded)).expanduser().exists():
        return
    digest = getattr(cfg, 'frame_checkpoint_sha256', None)
    here = Path(checkpoint).expanduser().resolve().parent
    for candidate in (here/Path(recorded).name, here/DEFAULT_FRAME_NAME):
        if candidate.is_file() and candidate.resolve() != Path(checkpoint).expanduser().resolve() and (
                digest is None or file_sha256(candidate) == digest):
            bind_frame_checkpoint(cfg, candidate)
            return
    raise SystemExit(f'The checkpoint uses the learned frame model {recorded}, which is not on this machine: pass '
                     f'--frame-checkpoint, or put it next to the checkpoint as {Path(recorded).name} or '
                     f'{DEFAULT_FRAME_NAME} (sha256 {digest})')


def configure_volume(args, spec, model_cfg):
    """Apply the CT and frame-model options (``args`` may lack any of them) to the checkpoint's volume spec and
    model configuration, in place."""
    get = lambda key: getattr(args, key, None)
    resolve_frame_checkpoint(model_cfg, args.checkpoint, get('frame_checkpoint'))
    for key in ('ct_level', 'ct_grid_scale', 'grid_scale'):
        if get(key) is not None:
            setattr(spec, key, get(key))
    if get('ct'):
        spec.ct_zarr = get('ct')
    if get('cache_dir'):
        spec.cache_dir = get('cache_dir')
    elif str(spec.ct_zarr).startswith(('s3://', 'http://', 'https://')) and not (
            spec.cache_dir and Path(spec.cache_dir).is_dir()):
        spec.cache_dir = default_cache_dir()
        print(f'CT chunk cache: {spec.cache_dir} (pass --cache-dir to choose another)', flush=True)


def use_gpu_crops(tracer, vol, device, slots):
    """Follower and frame-model crops from a GPU chunk cache (shared/crop_sampling/gpu.py; within ~3e-5 of the CPU
    sampler), and the frame model on the GPU: a copy, so a shared CPU predictor is unchanged."""
    import copy
    from vesuvius.neural_tracing.tracers.fibers.heading_model.model import HeadingPredictor
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.gpu import GPUChunkCache
    cache = GPUChunkCache(vol, device, slots=slots)
    tracer.chunk_cache = cache
    crops = lambda items, vol_, crop_, pool=None: cache.crops(items, vol_, crop_, pool)
    tracer.observations.crop_images = crops
    if tracer.frame_predictor is not None:
        tracer.frame_predictor = HeadingPredictor(copy.deepcopy(tracer.frame_predictor.model), device, crops)
    return cache
