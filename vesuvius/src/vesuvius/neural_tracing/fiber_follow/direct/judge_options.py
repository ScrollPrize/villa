"""Shared explicit development options and checkpoint reconstruction."""
import argparse
from .judge_slices import SliceConfig
from .judge_model import CTJudge, JudgeConfig, JUDGE_ARCHITECTURE
from .judge_policy import JudgePolicyConfig


def add_judge_options(ap):
    ap.add_argument('--judge', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--judge-ct', help='Judge CT store (default: the follower --ct store)')
    ap.add_argument('--judge-ct-level', type=int, default=0)
    ap.add_argument('--judge-ct-grid-scale', type=float,
                    help='Base voxels per judge CT voxel (default: follower scale when using the same store and level)')
    ap.add_argument('--judge-ct-origin', type=float, nargs=3, default=(0., 0., 0.))
    ap.add_argument('--judge-ct-cache')
    ap.add_argument('--judge-pixels', type=int, default=129)
    ap.add_argument('--judge-pixel-spacing', type=float,
                    help='Trace voxels per pixel (default: one CT voxel; finer spacing is rejected)')
    ap.add_argument('--judge-path-step', type=float, default=4.)
    ap.add_argument('--judge-history-length', type=float, default=128.)
    ap.add_argument('--judge-references', type=int, default=4)
    ap.add_argument('--judge-view-batch', type=int, default=16)
    ap.add_argument('--judge-loss-weight', type=float, default=.5)
    ap.add_argument('--judge-synthetic-fraction', type=float, default=.25)
    ap.add_argument('--judge-departed-fraction', type=float, default=0.,
                    help='Extra departed DAgger judge sequences per follower sample (0.5 requests 12 for batch 24)')
    ap.add_argument('--judge-accept', type=float, default=.9)
    ap.add_argument('--judge-alarm', type=float, default=.5)
    ap.add_argument('--judge-provisional', type=float, default=32.)


def configs(args, volume):
    source = args.judge_ct or volume.ct_zarr
    scale = args.judge_ct_grid_scale
    if scale is None:
        if source != volume.ct_zarr or args.judge_ct_level != volume.ct_level:
            raise ValueError('A different judge CT source or level requires --judge-ct-grid-scale')
        scale = volume.ct_grid_scale
    slices = SliceConfig(source=source, level=args.judge_ct_level, grid_scale=scale, trace_scale=volume.grid_scale,
                         origin=tuple(args.judge_ct_origin), cache=args.judge_ct_cache, pixels=args.judge_pixels,
                         spacing=args.judge_pixel_spacing, path_step=args.judge_path_step,
                         history_length=args.judge_history_length, references=args.judge_references)
    model = JudgeConfig(pixels=slices.pixels, spacing=slices.spacing, view_batch=args.judge_view_batch)
    policy = JudgePolicyConfig(accept=args.judge_accept, alarm=args.judge_alarm, provisional=args.judge_provisional)
    return slices, model, policy


def load_judge(ck, device):
    if ck.get('judge_architecture') != JUDGE_ARCHITECTURE:
        raise ValueError('Checkpoint has no compatible CT judge')
    model = CTJudge(JudgeConfig(**ck['judge_cfg'])).to(device)
    model.load_state_dict(ck['judge_ema'])
    model.eval().requires_grad_(False)
    slices = SliceConfig(**ck['judge_slices'])
    if ck.get('judge_source_sha256') is not None and slices.open().identity != ck['judge_source_sha256']:
        raise ValueError('Checkpoint native CT source metadata changed')
    return dict(judge=model, judge_slices=slices,
                judge_policy=JudgePolicyConfig(**ck['judge_policy']))
