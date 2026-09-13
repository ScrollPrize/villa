import argparse
from pathlib import Path

import cv2

from pgs_recon.utils import educelab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input image file path')
    parser.add_argument('--output-file', '-o',
                        help='Output mask image path. If not provided, will '
                             'default to {input file}_mask.png')
    parser.add_argument('--open-iterations', type=int, default=4,
                        help='The number of morphological open operations to '
                             'apply to the thresholded image')
    parser.add_argument('--debug', action='store_true',
                        help='If provided, save intermediate images for '
                             'debugging mask generation')
    args = parser.parse_args()

    # Load img
    img = cv2.imread(args.input_file)
    print(f'Loaded image: {img.shape}')

    # Generate mask
    print('Generating mask...')
    mask = educelab.generate_tray_mask(img,
                                       open_iterations=args.open_iterations,
                                       save_debug=args.debug)

    # Save image mask
    if args.output_file is None:
        input_file = Path(args.input_file)
        name = input_file.stem + '_mask.png'
        args.output_file = str(input_file.parent / name)
    print('Saving mask...')
    cv2.imwrite(args.output_file, mask)
    print('Done.')

if __name__ == '__main__':
    main()