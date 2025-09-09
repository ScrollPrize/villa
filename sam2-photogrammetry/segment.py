# Giorgio Angelotti - 2024

import argparse
import numpy as np
import torch
import gc
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# if using Apple MPS, fall back to CPU for unsupported ops
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def create_custom_palette():
    """
    Creates a 256-color palette where:
    - 0 = Black
    - 1 = White
    - 2 = Red
    - 3 = Blue
    - 4 = Green
    - 5+ = Colors from Matplotlib tab10
    """
    palette = np.zeros((256, 3), dtype=np.uint8)  # Initialize (256 colors, 3 channels)

    # Manually assign first five colors
    palette[0] = [0, 0, 0]        # Black
    palette[1] = [255, 255, 255]  # White
    palette[2] = [255, 0, 0]      # Red
    palette[3] = [0, 0, 255]      # Blue
    palette[4] = [0, 255, 0]      # Green

    # Load tab10 colormap from Matplotlib (10-class categorical colors)
    tab10 = plt.get_cmap("tab10").colors  # Returns 10 RGB values in range [0, 1]

    # Convert tab10 colors to 8-bit (0-255) and assign from index 5 onwards
    for i in range(5, min(15, 256)):  # Only assign up to 15 colors to avoid overflow
        palette[i] = np.array(tab10[i - 5]) * 255

    # Flatten the palette to a 768-length list (256 colors * 3 channels)
    return palette.flatten().tolist()

def save_segmented_mask(mask, mask_path):
    """
    Saves a segmentation mask using the custom palette.
    - `mask`: 2D NumPy array (H, W) with label indices.
    - `mask_path`: Path to save the mask as PNG.
    """
    # Convert to a PIL Image with mode 'P' (palette-based)
    segmentation_image = Image.fromarray(mask, mode='P')

    # Apply the custom palette
    segmentation_image.putpalette(create_custom_palette())

    # Save as PNG
    segmentation_image.save(mask_path)

def extract_digits(filename_stem):
    # Look for 1+ digits in the string
    match = re.search(r'(\d+)', filename_stem)
    if match:
        return int(match.group(1))
    return -1  # or return None, or some fallback

def find_jpg_enhanced_dirs(root_dir: Path):
    """
    Recursively search `root_dir` for all folders named 'JPGEnhanced'.
    Returns a list of Path objects.
    """
    # We look for any directory named "JPG_Enhanced" underneath `root_dir`
    return [p for p in root_dir.rglob('*') if p.is_dir() and p.name == "JPGEnhanced"]


def process_video_dir(video_dir: Path, mask_generator, predictor, recompute=False):
    """
    For a given folder `video_dir` containing frames (JPG/JPEG),
    run the video segmentation pipeline and save the results in a
    sibling 'Masks' folder (i.e., at the same level as `video_dir`).

    If all masks corresponding to the JPG files are found in the 'Masks' folder
    and `recompute` is False, this function will skip processing.
    """

    # Gather frame names (Path objects)
    frame_files = [
        f for f in video_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg"]
    ]
    # Sort frames by integer value of the filename (assuming <frame_index>.jpg)
    frame_files.sort(key=lambda f: extract_digits(f.stem))

    if not frame_files:
        print(f"No JPG files found in {video_dir}, skipping...")
        return

    # Check if all corresponding masks exist, and skip if so (unless recompute=True)
    mask_output_dir = video_dir.parent / "Masks"
    all_masks_exist = True
    for frame_file in frame_files:
        mask_filename = f"{frame_file.stem}_mask.png"
        if not (mask_output_dir / mask_filename).exists():
            all_masks_exist = False
            break

    if all_masks_exist and not recompute:
        print(f"All masks for {video_dir} are present. Skipping segmentation.")
        return

    # Generate a colormap from matplotlib (use 'jet', 'viridis', etc.)
    colormap = plt.get_cmap("tab10")  # or "viridis", "plasma", "jet", etc.

    # Convert colormap to a 256-entry palette (flatten to RGB triplets)
    palette = (np.array(colormap.colors) * 255).astype(np.uint8).flatten()

    # Pad the palette to 256 entries (since tab10 has only 10 colors)
    palette = np.tile(palette, 25)[:768]

    # Load one frame to generate initial masks using the automatic mask generator
    frame_idx = 0
    image_path = frame_files[frame_idx]
    image_pil = Image.open(image_path)
    image_grayscale = np.array(image_pil.convert("L"))
    image_rgb = np.array(image_pil.convert("RGB"))

    # Generate initial masks
    masks = mask_generator.generate(image_rgb)

    # Create a list with average pixel values per segment in the grayscale image
    darkness = []
    for m in masks:
        darkness.append(np.mean(image_grayscale[m["segmentation"]]))

    # Order the masks by brightness (darkest -> brightest)
    sorted_indices = np.argsort(darkness)
    masks = [masks[i] for i in sorted_indices]

    # Initialize the video predictor's internal state
    inference_state = predictor.init_state(video_path=str(video_dir))

    # The frame index where we add the initial annotation
    ann_frame_idx = 0

    # Add each mask except the last
    for ann_obj_id in range(len(masks)):
        _, _, _, = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id+1,
            mask=masks[ann_obj_id]['segmentation'],
        )

    # Run propagation throughout the "video"
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(out_obj_ids)
        }

    # Create output Masks folder (sibling to `video_dir`)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    # Save masks for each frame
    for out_frame_idx, frame_file in tqdm(enumerate(frame_files), desc="saving masks", total=len(frame_files)):
        frame = np.array(Image.open(frame_file))
        height, width = frame.shape[:2]

        # Create an empty mask with background set to 0
        mask = np.zeros((height, width), dtype=np.uint8)

        # If we have segmentation data for this frame, set the segment IDs
        if out_frame_idx in video_segments:
            for obj_id, out_mask in video_segments[out_frame_idx].items():
                mask[out_mask[0]] = obj_id

        # Save mask as <filename>_mask.png
        base_filename = frame_file.stem
        mask_filename = f"{base_filename}_mask.png"
        mask_path = mask_output_dir / mask_filename

        save_segmented_mask(mask, mask_path)

        predictor.reset_state(inference_state)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process multiple scroll/orientation folders containing JPGEnhanced subfolders."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the root directory containing scroll/orientation folders."
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_tiny.pt",
        help="Relative or absolute path to the SAM2 checkpoint."
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="/sam2_hiera_t.yaml",
        help="Relative or absolute path to the model configuration file."
    )
    parser.add_argument(
        "--photo_t_checkpoint",
        type=str,
        default="./checkpoints/photo2_ruler_t_1000.torch",
        help="Relative or absolute path to the photo_t checkpoint for partial model weights."
    )
    # New argument: if set, force recomputation
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all masks even if they already exist."
    )

    args = parser.parse_args()

    # Convert input paths to Path objects and then resolve them
    root_dir = Path(args.root_dir).resolve()
    sam2_checkpoint = Path(args.sam2_checkpoint).resolve()
    model_cfg = Path(args.model_cfg).resolve()
    photo_t_checkpoint = Path(args.photo_t_checkpoint).resolve()

    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire script
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # Build the main model
    sam2 = build_sam2(
        str(model_cfg),              # pass string to builder
        str(sam2_checkpoint),        # pass string to builder
        device=device,
        apply_postprocessing=False,
    )

    # Build the mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=64,
        points_per_batch=16,
        min_mask_region_area=2048,
        use_m2m=True
    )

    # Build the video predictor
    predictor = build_sam2_video_predictor(
        str(model_cfg),
        str(sam2_checkpoint),
        non_overlap_masks=True,
        device=device
    )

    # Load partial model weights
    predictor_weights = torch.load(str(photo_t_checkpoint), map_location=device, weights_only=True)
    mask_generator.predictor.model.load_state_dict(predictor_weights, strict=False)
    predictor.load_state_dict(predictor_weights, strict=False)

    # Find all JPG_Enhanced subfolders
    jpg_enhanced_dirs = find_jpg_enhanced_dirs(root_dir)
    if not jpg_enhanced_dirs:
        print("No 'JPGEnhanced' folders found under root_dir. Exiting.")
        return

    # Process each JPGEnhanced folder
    for jpg_enhanced_dir in jpg_enhanced_dirs:
        print(f"\nProcessing folder: {jpg_enhanced_dir}")
        process_video_dir(
            video_dir=jpg_enhanced_dir,
            mask_generator=mask_generator,
            predictor=predictor,
            recompute=args.recompute
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
