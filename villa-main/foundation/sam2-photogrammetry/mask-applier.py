import os
import argparse
import numpy as np
import piexif
from PIL import Image
import multiprocessing
from tqdm import tqdm

def apply_mask(image_path, mask_folder, output_folder, recompute):
    """
    Given:
      - image_path: Full path to a .jpg file in a "JPG" or "JPGEnhanced" folder.
      - mask_folder: Path to the corresponding "Masks" folder.
      - output_folder: Path to the corresponding "Masked" folder.
      - recompute: If True, overwrite existing masked images.

    The mask is expected to have the same basename, with suffix _mask.png or _mask_strange-cc.png.
    Only label==1 is retained in the masked jpg; additionally a _masked_mask.png is saved with label 1 white and others black.
    """
    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{base_name}_masked.jpg")

    # Check if output already exists
    if os.path.exists(output_path) and not recompute:
        return  # Skip if recompute is False

    # Possible mask filenames
    possible_masks = [
        f"{base_name}_mask.png",
        f"{base_name}_mask_strange-cc.png",
    ]

    mask_path = None
    for pm in possible_masks:
        candidate = os.path.join(mask_folder, pm)
        if os.path.isfile(candidate):
            mask_path = candidate
            break

    # If no corresponding mask found, skip
    if mask_path is None:
        return

    try:
        # Load the JPG
        with Image.open(image_path) as img:
            img_array = np.array(img)

        # Load the mask (ensure it is in mode "P" so that the palette is used)
        with Image.open(mask_path) as msk:
            if msk.mode != "P":
                msk = msk.convert("P")
            mask_array = np.array(msk, dtype=np.uint8)

        # Check dimension match
        if img_array.shape[:2] != mask_array.shape:
            print(f"[WARN] Dimension mismatch: {filename} vs. {os.path.basename(mask_path)}. Skipping.")
            return

        # Determine which pixels to keep (where mask==1)
        keep_pixels = (mask_array == 1)
        img_array[~keep_pixels] = [0, 0, 0]

        # --- Create and save the masked mask image ---
        # Build a binary mask: pixels with label 1 become 1, all others 0.
        binary_mask = keep_pixels.astype(np.uint8)
        # Create a new palettized image
        mask_img = Image.fromarray(binary_mask, mode="P")
        # Create a two-color palette: index 0->black, index 1->white.
        # (A palette must be 768 values long: 256 RGB triples.)
        palette = [0, 0, 0, 255, 255, 255] + [0] * (768 - 6)
        mask_img.putpalette(palette)
        # Define output path for the masked mask PNG
        output_mask_path = os.path.join(output_folder, f"{base_name}_masked_mask.png")
        mask_img.save(output_mask_path, "PNG")
        # --- End of masked mask creation ---

        # Save the masked image (without EXIF initially)
        result_img = Image.fromarray(img_array, mode="RGB")
        temp_output_path = output_path + "_temp.jpg"
        result_img.save(temp_output_path, "JPEG", subsampling=0, quality=100)

        # Transfer EXIF metadata from original to masked image
        piexif.transplant(image_path, temp_output_path)

        # Rename the final image
        os.rename(temp_output_path, output_path)

    except Exception as e:
        print(f"[ERROR] Failed on {filename}: {e}")


def worker_func(task):
    """
    A simple top-level wrapper so that `pool.imap_unordered(worker_func, tasks)`
    can pickle and call this function. 
    `task` is a tuple: (image_path, mask_folder, masked_folder, recompute)
    """
    (image_path, mask_folder, masked_folder, recompute) = task
    return apply_mask(image_path, mask_folder, masked_folder, recompute)


def find_folders(root_dir):
    """
    Recursively search root_dir for any folder named 'JPG' or 'JPGEnhanced'.
    If found, check if there's a sibling folder named 'Masks'.
    Yield a tuple: (image_folder_path, mask_folder_path, masked_folder_path).
    """
    matches = []
    for current_root, dirs, _ in os.walk(root_dir):
        folder_name = os.path.basename(current_root)
        if folder_name in ["JPG", "JPGEnhanced"]:
            parent = os.path.dirname(current_root)
            mask_folder = os.path.join(parent, "Masks")
            if os.path.isdir(mask_folder):
                masked_folder = os.path.join(os.path.dirname(parent), "Masked")
                matches.append((current_root, mask_folder, masked_folder))
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Traverse a root directory, auto-discover pairs of JPG/JPGEnhanced+Masks, and produce Masked output."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Root directory where we recursively look for subfolders named 'JPG' or 'JPGEnhanced' (alongside 'Masks').")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel worker processes to use (default: all CPUs).")
    parser.add_argument("--recompute", action="store_true", help="Recompute and overwrite existing masked JPGs.")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    workers = args.workers
    recompute = args.recompute

    # Find all relevant folder triples.
    folder_triples = find_folders(root_dir)
    if not folder_triples:
        print(f"No 'JPG' or 'JPGEnhanced' folders with sibling 'Masks' found under {root_dir}. Exiting.")
        return

    # Build a list of tasks: (image_path, mask_folder, masked_folder, recompute).
    tasks = []
    valid_exts = (".jpg", ".jpeg")
    for (img_folder, msk_folder, out_folder) in folder_triples:
        os.makedirs(out_folder, exist_ok=True)  # create Masked folder if missing

        # Gather .jpg / .jpeg images in discovered folder
        for fname in os.listdir(img_folder):
            if os.path.splitext(fname.lower())[-1] in valid_exts:
                image_path = os.path.join(img_folder, fname)
                tasks.append((image_path, msk_folder, out_folder, recompute))

    if not tasks:
        print("No .jpg or .jpeg files found in discovered folders. Exiting.")
        return

    # Use a process pool with a top-level worker_func to avoid pickling issues
    with multiprocessing.Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker_func, tasks),
                      total=len(tasks), desc="Processing", unit="image"):
            pass

    print("Done.")


if __name__ == "__main__":
    main()
