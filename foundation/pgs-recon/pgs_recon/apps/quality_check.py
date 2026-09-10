import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import blur_effect
from tqdm import tqdm

from pgs_recon.utils import quality


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgs-dir', '-p', required=True,
                        help='PGS Scan directory')
    parser.add_argument('--save-plots', action='store_true')
    args = parser.parse_args()

    # Load scan metadata
    scan_dir = Path(args.pgs_dir)
    meta_path = scan_dir / 'metadata.json'
    with meta_path.open() as f:
        meta = json.loads(f.read())

    # setup data structure
    num_cam = len(meta['scanner']['cameras'])
    num_pos = len(meta['scan']['capture_positions'])
    num_cap = len(meta['scan']['capture_settings'])
    labels = [c['desc'] for c in meta['scanner']['cameras']]
    metrics = np.full((num_cam, num_pos, num_cap, 2), np.nan)

    # Get file name info
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()
    images = list(scan_dir.glob(f'{prefix}*.{ext}'))
    images.sort()

    # calculate the metrics
    dynamic_range = None
    for p in tqdm(images, 'Calculating metrics'):
        # parse the image indices
        indices = p.name.removeprefix(prefix).removesuffix(f'.{ext}').split('_')
        cam, pos, cap = [int(idx) for idx in indices]

        # Load the image
        img = iio.imread(p)

        # update the global dynamic range
        high = None
        if img.dtype == np.uint8:
            high = 2 ** 8 - 1
        elif img.dtype == np.uint16:
            high = 2 ** 16 - 1
        if high is not None:
            if dynamic_range is None:
                dynamic_range = [0, high]
            else:
                dynamic_range[1] = max(dynamic_range[1], 255)

        # (0) Calculate the mean brightness
        metrics[cam, pos, cap, 0] = quality.measure_exposure(img)

        # (1) Calculate the blur effect
        metrics[cam, pos, cap, 1] = blur_effect(img)

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2))
    fig.suptitle(scan_dir.name)

    axs[0, 0].set_title('exposure (regular)')
    axs[0, 0].set_xlabel('capture position')
    axs[0, 0].set_ylabel('exposure')
    axs[0, 0].set_ylim(dynamic_range)

    axs[0, 1].set_title('exposure (IR)')
    axs[0, 1].set_xlabel('capture position')
    axs[0, 1].set_ylabel('exposure')
    axs[0, 1].set_ylim(dynamic_range)

    axs[1, 0].set_xlabel('capture position')
    axs[1, 0].set_ylabel('blur effect')
    axs[1, 0].set_ylim([0., 1.])

    axs[1, 1].axis('off')

    clf = neighbors.LocalOutlierFactor(n_neighbors=6)
    outlier_kwargs = {'marker': 'X', 'mfc': 'red', 'mec': 'none'}
    for cam in range(num_cam):
        for cap in range(num_cap):
            # style for caps
            label = f'[{cam}] {labels[cam]}'

            # exposure
            val = metrics[cam, :, cap, 0]
            if not np.isnan(val).all():
                x = np.nan_to_num(val)
                x = x.reshape(1, -1)
                y, clf = quality.detect_outliers(x, clf=clf)
                markers_on = np.where(y == -1)[0]
                scores = clf.negative_outlier_factor_[markers_on]
                if len(markers_on) > 0:
                    outliers = ', '.join(
                        f'[{i}]({s:.3g})' for i, s in zip(markers_on, scores))
                    print(f'[WARNING] [{cam}.{cap}] Detected {len(markers_on)} '
                          f'exposure outliers: {outliers}')
                axs[0, cap].plot(val, label=label, markevery=markers_on,
                                 **outlier_kwargs)

            # blur
            val = metrics[cam, :, cap, 1]
            if not np.isnan(val).all():
                axs[1, 0].plot(val, label=label)

    # Hide empty subplots
    for r in axs:
        for a in r:
            if a.lines:
                a.legend()
            else:
                a.set_visible(False)

    # finalize layout
    plt.tight_layout()

    if args.save_plots:
        plt.savefig(f'{scan_dir.name}-metrics.png')
    else:
        plt.show()


if __name__ == '__main__':
    main()