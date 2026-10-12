"""What the tracing CLIs (inference.py, volume.py, service.py) share: the checkpoint, CT and frame-model options and
their resolution on this machine, and GPU crops.

Checkpoint (``--checkpoint``): a local file, or a Hugging Face model repo ``ORG/NAME`` (also ``hf://ORG/NAME``, with
``@REVISION`` for a branch, tag or commit; default ``DEFAULT_CHECKPOINT``), downloaded into the Hugging Face cache
(``$HF_HOME``). The repo holds the follower checkpoint (``ckpt_final.pt``, else its only non-frame ``.pt``, e.g. ``last.pt``) and its
frame model (``heading_model.pt`` or ``frame_model.pt``) next to it.

A checkpoint records where it was trained: its CT (``vol_spec``: store, level, scales, chunk cache folder) and its
learned frame model (an absolute path pinned by sha256). On another machine:
- CT: ``--ct``/``--ct-level``/``--ct-grid-scale``/``--grid-scale`` override the recorded store and scales. A remote
  store (s3://, https://) needs a persistent chunk cache: ``--cache-dir``, else the recorded folder when it exists here,
  else ``$XDG_CACHE_HOME/vesuvius/volume_cache`` (``~/.cache/...``).
- Frame model: ``--frame-checkpoint``, else the recorded path (relocated by ``TRACERS_PATH_MAP``) when it exists, else
  a file next to the follower checkpoint: the recorded file name, then ``frame_model.pt``, ``heading_model.pt``. A found
  file must have the recorded sha256.
"""
from __future__ import annotations

import os
from pathlib import Path

from vesuvius.neural_tracing.tracers.shared.hub import checkpoint_source, hf_repo, hub_folder  # noqa: F401 (re-exported)

DEFAULT_CHECKPOINT = 'scrollprize/fiber_single_small'
FOLLOWER_NAME = 'ckpt_final.pt'
FRAME_NAMES = ('frame_model.pt', 'heading_model.pt')


def add_checkpoint_argument(ap):
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                    help=f'follower checkpoint: a file, or a Hugging Face repo ORG/NAME[@REVISION] holding it and its '
                         f'frame model (default: {DEFAULT_CHECKPOINT})')


def resolve_checkpoint(checkpoint):
    """The local file of ``checkpoint`` (a file, or a Hugging Face repo, downloaded with its frame model)."""
    repo = hf_repo(checkpoint)
    if repo is None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise SystemExit(f'Checkpoint not found: {checkpoint}')
        return str(path)
    repo_id = repo[0]
    folder = hub_folder(repo)
    if (folder/FOLLOWER_NAME).is_file():
        return str(folder/FOLLOWER_NAME)
    followers = [f for f in sorted(folder.glob('*.pt')) if f.name not in FRAME_NAMES]
    if len(followers) != 1:
        raise SystemExit(f'Hugging Face repo {repo_id} needs {FOLLOWER_NAME}, or exactly one checkpoint besides its '
                         f'frame model ({", ".join(FRAME_NAMES)}); it has {[f.name for f in followers]}')
    return str(followers[0])


def add_volume_arguments(ap):
    ap.add_argument('--ct', help="CT zarr (local path or s3://); default: the checkpoint's")
    ap.add_argument('--ct-level', type=int, help="CT pyramid level read; default: the checkpoint's")
    ap.add_argument('--ct-grid-scale', type=float, help='base voxels per CT voxel at that level (normally 2**level)')
    ap.add_argument('--grid-scale', type=float, help="base voxels per trace-grid voxel; default: the checkpoint's")
    ap.add_argument('--cache-dir', help='local cache for remote CT chunks (default: the recorded folder when it exists, '
                                        'else $XDG_CACHE_HOME/vesuvius/volume_cache)')
    ap.add_argument('--frame-checkpoint', help='learned frame model (default: the recorded path, else the recorded '
                                               f'file name, {" or ".join(FRAME_NAMES)} next to the checkpoint)')


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
    here = Path(checkpoint).expanduser().absolute().parent  # not resolved: Hugging Face cache files are symlinks
    for candidate in (here/Path(recorded).name, *(here/name for name in FRAME_NAMES)):
        if candidate.is_file() and candidate.resolve() != Path(checkpoint).expanduser().resolve() and (
                digest is None or file_sha256(candidate) == digest):
            bind_frame_checkpoint(cfg, candidate)
            return
    raise SystemExit(f'The checkpoint uses the learned frame model {recorded}, which is not on this machine: pass '
                     f'--frame-checkpoint, or put it next to the checkpoint as {Path(recorded).name} or '
                     f'{" or ".join(FRAME_NAMES)} (sha256 {digest})')


def configure_volume(args, spec, model_cfg, checkpoint):
    """Apply the CT and frame-model options (``args`` may lack any of them) to the volume spec and model configuration
    of the checkpoint file ``checkpoint``, in place."""
    get = lambda key: getattr(args, key, None)
    resolve_frame_checkpoint(model_cfg, checkpoint, get('frame_checkpoint'))
    for key in ('ct_level', 'ct_grid_scale', 'grid_scale'):
        if get(key) is not None:
            setattr(spec, key, get(key))
    if get('ct'):
        spec.ct_zarr = get('ct')
    if get('cache_dir'):
        spec.cache_dir = get('cache_dir')
    elif os.environ.get('VESUVIUS_CT_READER') != 'vc3d' and str(spec.ct_zarr).startswith(('s3://', 'http://', 'https://')) and not (
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
