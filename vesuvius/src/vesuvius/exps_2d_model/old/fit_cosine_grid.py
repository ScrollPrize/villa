import math
from pathlib import Path
 
import tifffile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import load_unet
import cv2

from fit import fit_cosine_grid



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("Fit 2D cosine grid to an image or tiled UNet outputs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to input TIFF image (stack) or directory containing precomputed "
            "tiled UNet outputs (_cos/_mag/_dir0/_dir1)."
        ),
    )
    parser.add_argument(
        "--stages-json",
        type=str,
        default="default_stages.json",
        help="Path to stages/weights JSON (see default_stages.json).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--grid-step",
        type=int,
        default=4,
        help="Vertical coarse grid step in sample-space pixels for the internal eval grid.",
    )
    parser.add_argument(
        "--output-scale",
        type=int,
        default=4,
        help="Integer scale factor for saving reconstructions (snapshots and final).",
    )
    parser.add_argument(
        "--cosine-periods",
        type=float,
        default=32.0,
        help="Number of cosine periods across the sample-space width.",
    )
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help="Global multiplier for internal sample-space resolution (applied after x/y base sizing).",
    )
    parser.add_argument(
        "--samples-per-period",
        type=float,
        default=1.0,
        help="Number of coarse grid steps per cosine period horizontally.",
    )
    parser.add_argument(
        "--dense-samples-per-period",
        type=float,
        default=8.0,
        help="Dense samples per cosine period for the internal x resolution.",
    )
    parser.add_argument(
        "--img-downscale-factor",
        type=float,
        default=4.0,
        help="Downscale factor for internal resolution relative to avg image size.",
    )
    parser.add_argument(
        "--cos-mask-periods",
        type=float,
        default=5.0,
        help="Number of cosine periods from the left edge of sample space used for the loss mask.",
    )
    parser.add_argument(
        "--cos-mask-v-extent",
        type=float,
        default=0.1,
        help="Vertical half-extent in normalized sample-space v (in [0,1]) of the cosine loss band (default: 0.1).",
    )
    parser.add_argument(
        "--cos-mask-v-ramp",
        type=float,
        default=0.05,
        help="Vertical ramp width in normalized sample-space v for the cosine loss band (linear fade-out to 0 outside the band).",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to UNet checkpoint (.pt). If set, run UNet on the specified TIFF "
            "layer from --input (when it is a TIFF file) and fit the cosine grid "
            "to its channel-0 output. Must not be set when --input is a directory "
            "of precomputed tiled UNet outputs."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "Layer index of the input TIFF stack (when --input is a file), or the "
            "layer suffix to select in a directory of tiled UNet outputs "
            "(filenames of the form *_layerXXXX_{cos,mag,dir0,dir1}.tif)."
        ),
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("CX", "CY"),
        default=None,
        help="Mask center in pixels (CX CY), 0,0 = top-left; default is image center.",
    )
    parser.add_argument(
        "--unet-crop",
        type=int,
        default=16,
        help="Pixels to crop from each image border after UNet inference, before downscaling (only used with --unet-checkpoint).",
    )
    parser.add_argument(
        "--min-dx-grad",
        type=float,
        default=0.03,
        help="Minimum gradient of rotated x-coordinate along coarse x (frequency lower bound).",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile CosineGridModel with torch.compile (PyTorch 2.x) for faster training.",
    )
    parser.add_argument(
        "--final-float",
        action="store_true",
        help="Save final recon/gt/modgt as float32 TIFFs instead of uint8.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Optional crop rectangle in pixels (x,y,w,h) applied before fitting.",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help="If set > 0 and output-prefix is given, save a reconstruction snapshot every N steps.",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="If set, snapshot additional debug outputs (loss mask and diff) alongside reconstructions.",
    )
    parser.add_argument(
        "--for-video",
        action="store_true",
        help=(
            "Video mode: disable output upscaling, draw the grid on a black "
            "background, save masks as JPG, and use LZW compression for TIFFs."
        ),
    )
    parser.add_argument(
        "--use-image-mask",
        action="store_true",
        help="Enable image-space Gaussian loss mask in stage 3 in addition to the cosine-domain mask.",
    )
    args = parser.parse_args()
    fit_cosine_grid(
        image_path=args.input,
        lr=args.lr,
        grid_step=args.grid_step,
        stages_json=args.stages_json,
        min_dx_grad=args.min_dx_grad,
        device=args.device,
        output_prefix=args.output_prefix,
        snapshot=args.snapshot,
        output_scale=args.output_scale,
        dbg=args.dbg,
        mask_cx=(args.center[0] if args.center is not None else None),
        mask_cy=(args.center[1] if args.center is not None else None),
        cosine_periods=args.cosine_periods,
        sample_scale=args.sample_scale,
        samples_per_period=args.samples_per_period,
        dense_samples_per_period=args.dense_samples_per_period,
        img_downscale_factor=args.img_downscale_factor,
        for_video=args.for_video,
        unet_checkpoint=args.unet_checkpoint,
        unet_layer=args.layer,
        unet_crop=args.unet_crop,
        crop=tuple(args.crop) if args.crop is not None else None,
        compile_model=args.compile_model,
        final_float=args.final_float,
        cos_mask_periods=args.cos_mask_periods,
        cos_mask_v_extent=args.cos_mask_v_extent,
        cos_mask_v_ramp=args.cos_mask_v_ramp,
        use_image_mask=args.use_image_mask,
    )


if __name__ == "__main__":
    main()
