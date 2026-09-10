import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import configargparse
import imageio.v3 as iio
import numpy as np
from educelab import imgproc
from educelab.imgproc import pipeline
from skimage import img_as_float
from tqdm import tqdm

from pgs_recon.utility import run_command


def write_config(args, config_path=None):
    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d_%H%M%S')
    args.name = datetime_str + '_' + str(Path(args.input).stem)

    # Write config after all arguments have been changed
    if config_path is None:
        config_path = Path(
            args.output) / f'{datetime_str}_{args.name}_convert_config.txt'
    args.config = str(config_path)
    with config_path.open(mode='w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            arg = arg.replace('_', '-')
            file.write(f'{arg} = {attr}\n')


def has_group_opt(args, grp):
    return any([getattr(args, b.dest) is not None for b in grp._group_actions])


def main():
    parser = configargparse.ArgumentParser(prog='pgs-convert')
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PGS dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PGS dataset directory')
    parser.add_argument('--log-level', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='WARNING', help='Logging level')

    convert_opts = parser.add_argument_group('conversion options')
    convert_opts.add_argument('--file-type', '-f', choices=['jpg', 'tif'],
                              default='jpg', type=str.lower,
                              help='Output image format')
    convert_opts.add_argument('--if-same-type', default='copy',
                              choices=['skip', 'copy', 'convert'],
                              help='Behavior to use when the input file type '
                                   'matches the target: (skip) the dataset, '
                                   '(copy) directly to the output directory, '
                                   '(convert) files anyway. Files are always '
                                   'converted if one of the enhancement '
                                   'options is provided.')
    convert_opts.add_argument('--force-copy', default=False,
                              action=argparse.BooleanOptionalAction,
                              help='When performing a dataset copy, ignore '
                                   'files which would be overwritten in the '
                                   'output directory')
    convert_opts.add_argument('--quality', '-q', type=int,
                              help='Output image quality. Range depends on '
                                   '--file-type')

    file_opts = parser.add_argument_group('file filter options')
    file_opts.add_argument('--filter-cam', type=int, metavar='INT',
                           help='Filter by camera index')
    file_opts.add_argument('--filter-pos', type=int, metavar='INT',
                           help='Filter by position index')
    file_opts.add_argument('--filter-cap', type=int, metavar='INT',
                           help='Filter by capture index')

    perf_opts = parser.add_argument_group('performance options')
    perf_opts.add_argument('--threads', '-t', type=int,
                           help="Maximum number of threads to use when "
                                "converting images")

    # add the enhancement pipeline options
    enhance_opts = pipeline.add_parser_enhancement_group(parser)

    # parse arguments and commands
    args = parser.parse_args()
    apply_pipeline, cmds = pipeline.parse_and_build(args.commands)

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('pgs-convert')

    # Validate the input directory
    scan_dir = Path(args.input)
    if not scan_dir.exists():
        logger.error(f'Input directory does not exist: {str(scan_dir)}')
        sys.exit(1)

    # Get output directory
    output_dir = Path(args.output)

    # Load the scan metadata
    meta_path = scan_dir / 'metadata.json'
    if not meta_path.exists():
        logger.error(f'File not found: {str(meta_path)}')
        sys.exit(1)
    with meta_path.open(encoding='utf-8') as f:
        meta = json.loads(f.read())

    # Get file name info
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()

    # Handle matching format
    fmt_match = ext == args.file_type
    has_enhance_opt = has_group_opt(args, enhance_opts)
    if fmt_match and not has_enhance_opt:
        # Format matches and not copying
        if args.if_same_type == 'skip':
            logger.info('Input dataset matches requested format. '
                        'Data will not be copied or converted.')
            sys.exit(0)

        # Format matches and copying directly
        elif args.if_same_type == 'copy':
            logger.info('Input dataset matches requested format. '
                        'Copying to the output directory.')
            try:
                shutil.copytree(scan_dir, output_dir,
                                dirs_exist_ok=args.force_copy)
            except FileExistsError as e:
                logger.error(e)
                sys.exit(1)
            write_config(args)
            sys.exit(0)

    # File filter
    cam_f = f'{args.filter_cam:03}' if args.filter_cam is not None else '*'
    pos_f = f'_{args.filter_pos:05}' if args.filter_pos is not None else '_*'
    cap_f = f'_{args.filter_cap:02}' if args.filter_cap is not None else '_*'
    suffix = f'{cam_f}{pos_f}{cap_f}'

    # Get a list of images
    images = list(scan_dir.glob(f'{prefix}{suffix}.{ext}'))
    images.sort()
    if len(images) == 0:
        logger.error('No images found in directory.')
        sys.exit(1)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # If we're on a SLURM node, os.cpu_count returns the hardware CPUs, which is
    # not necessarily what's available to the job. To avoid deadlocks, override
    # the default with what's actually usable.
    if args.threads is None and 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
        args.threads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    logger.debug(f'Max worker threads: {"auto" if args.threads is None else args.threads}')

    # Write config before convert
    write_config(args)

    # Modify the metadata and save to out dir
    meta['scan']['output_dir'] = str(output_dir.resolve())
    meta['scan']['format'] = args.file_type.upper()
    if len(cmds):
        meta['enhancements'] = cmds
    meta_path = output_dir / 'metadata.json'
    with meta_path.open('w', encoding='utf8') as f:
        json.dump(meta, f, indent=4)

    # Define conversion function
    def convert_image(p) -> bool:
        # Load image
        try:
            img = iio.imread(p)
        except ValueError:
            logger.error(f'Failed to load file: {str(p)}')
            return False
        in_dtype = img.dtype

        # Convert to float for processing
        img = img_as_float(img)

        # Process the image
        img = apply_pipeline(img)

        # Determine output format
        kwargs = {}

        # Type conversion
        if args.file_type == 'jpg':
            out_dtype = np.uint8
        else:
            out_dtype = in_dtype
        img = np.clip(img, 0., 1.)
        img = imgproc.as_dtype(img, out_dtype)

        # Format specific opts
        if args.file_type == 'jpg':
            kwargs[
                'quality'] = args.quality if args.quality is not None else 100
        elif args.file_type == 'tif':
            kwargs['compression'] = 'zlib'
            kwargs['compressionargs'] = {'level': 9}

        # Save the image to disk
        out_file = p.with_suffix(f'.{args.file_type}').name
        out_path = output_dir / out_file
        iio.imwrite(out_path, img, **kwargs)
        return True

    # Convert images (single-threaded)
    if args.threads == 1:
        results = []
        for p in tqdm(images, desc='Converting images'):
            results.append(convert_image(p))
    # Convert images (multithreaded)
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = executor.map(convert_image, images)
            results = list(
                tqdm(futures, total=len(images), desc='Converting images'))
    # Report success
    num_failed = results.count(False)
    if num_failed > 0:
        logger.warning(f'{num_failed} images failed to convert.')

    # Setup metadata copy
    cmd = ['exiftool', '-q', '-P', '-overwrite_original']

    # Skip tags that don't make sense in JPGs
    if args.file_type in ['jpg', 'png']:
        cmd.extend(['-XMP-tiff:all=', '-ExifIFD:BitsPerSample=',
                    '-IFD0:BitsPerSample='])

    # Original tags from original files
    # Use dummy _ to get OS separator then strip dummy _
    meta_dir = str(scan_dir / '_')[:-1]
    cmd.extend(['-TagsFromFile', f'{meta_dir}%f.{ext}'])

    # Map all the other tags
    cmd.append('-all:all')

    # Iterate over the output dir
    cmd.append(str(output_dir))

    # Copy metadata
    logger.info('Copying metadata...')
    logger.debug(f'Metadata args: {cmd}')
    run_command(cmd)


if __name__ == '__main__':
    main()
