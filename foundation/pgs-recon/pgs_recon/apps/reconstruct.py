"""Run the photogrammetry pipeline on a set of input images."""
import argparse
import atexit
import json
import logging
import sys
from datetime import datetime as dt, timezone as tz
from pathlib import Path

import exiftool
import configargparse
import sfm_utils as sfm

from pgs_recon.openmvg import (compute_features, compute_matches,
                               geometric_filter, init_sfm_generic,
                               mvg_colorize_sfm, mvg_compute_known, mvg_sfm,
                               mvg_autoscale, mvg_to_mvs)
from pgs_recon.openmvs import (mvs_densify, mvs_reconstruct, mvs_refine,
                               mvs_texture)
from pgs_recon.pgs_data import init_sfm_pgs, get_tag_option
from pgs_recon.utility import current_timestamp
from pgs_recon.utils.apps import setup_logging


def init_sfm_generic2(scan_dir: Path, sfm_file: Path, camdb_path: Path):
    """Init an SfM scene from the given directory"""
    logger = logging.getLogger(__name__)
    # Load the camera db
    cam_db = sfm.openmvg_load_camdb(camdb_path)

    # Get a list of files
    extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
    extensions = extensions.union([ext.upper() for ext in extensions])
    all_files = list(scan_dir.glob(f'*.*'))
    all_files.sort()
    images = []
    for f in all_files:
        if 'mask.png' in str(f):
            logger.warning(f'{str(f.name)} is a mask image')
        elif Path(f).suffix in extensions:
            images.append(f)
        else:
            logger.debug(f'Ignoring file: {str(f.name)}')
    if len(images) == 0:
        logger.error(
            'Provided scan metadata specifies file pattern, but no files match.')
        raise RuntimeError()

    # Get image metadata
    with exiftool.ExifToolHelper() as et:
        img_metadata = et.get_metadata([str(i) for i in images])

    # Setup sfm
    scene = sfm.Scene()
    scene.root_dir = scan_dir

    # Fill out sfm with data
    for img in images:
        # Lookup this images tags
        tags = next((i for i in img_metadata if i['File:FileName'] == img.name),
                    None)
        if tags is None:
            logger.error(f'No tags loaded for image: {str(img.name)}')
            continue

        # Setup view
        view = sfm.View()
        view.path = img
        view.width = get_tag_option(tags, ['File:ImageWidth', 'EXIF:ImageWidth'])
        view.height = get_tag_option(tags, ['File:ImageHeight', 'EXIF:ImageHeight'])
        view.make = tags['EXIF:Make']
        view.model = tags['EXIF:Model']

        # Setup intrinsic
        intrinsic = sfm.IntrinsicRadialK3()
        intrinsic.width = view.width
        intrinsic.height = view.height
        intrinsic.focal_length = tags['EXIF:FocalLength']
        if f'{view.make} {view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.make} {view.model}']
        elif f'{view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.model}']
        else:
            logger.warning(
                f'Camera not in database: {view.make} {view.model}. Ignoring file: {img.name}')
            continue

        # Only add everything to the SfM at the end
        scene.add_view(view)
        view.intrinsic = scene.add_intrinsic(intrinsic)
        view.pose = scene.add_pose(sfm.Pose())

    sfm.export_scene(path=sfm_file, scene=scene)


def main():
    parser = configargparse.ArgumentParser(prog='pgs-recon')

    # Generic options
    parser.add_argument('--config', '-c', is_config_file=True,
                        help='Config file path')
    parser.add_argument('--input', '-i', required=True,
                        help='directory of input images')
    parser.add_argument('--output', '-o', required=True,
                        help='directory for output files')
    parser.add_argument('--name', '-n', type=str, help='Experiment name.')
    parser.add_argument('--file-type', choices=['ply', 'obj'],
                        default='obj', type=str.lower,
                        help='Output format for final textured mesh')
    parser.add_argument('--focal-length', '-f', type=int, default=None,
                        help='focal length in pixels', metavar='n')
    parser.add_argument('--new-importer', default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--import-pgs-scan', '-p',
                        action=argparse.BooleanOptionalAction,
                        help='Input directory is assumed to be a PGS Scan '
                             'directory')
    parser.add_argument('--import-calib', type=str,
                        help='When importing a PGS Scan, merge provided PGS '
                             'calibration file with the imported camera '
                             'configurations.')
    parser.add_argument('--log-level', default='INFO', type=str.upper,
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument('--stop-after', choices=['sfm', 'autoscale'],
                        help='stop after producing a reusable SfM or scaled '
                             'SfM artifact')
    parser.add_argument('--resume-sfm', type=str,
                        help='skip OpenMVG and scaling, then continue OpenMVS '
                             'from this existing SfM .bin file')

    # Hidden opts
    parser.add_argument('--path', type=str, default='/usr/local/',
                        help=configargparse.SUPPRESS)
    parser.add_argument('--threads', type=int, help=configargparse.SUPPRESS)

    opts_desc = parser.add_argument_group('describer options')
    opts_desc.add_argument('--describer-method',
                           choices=['SIFT', 'AKAZE_FLOAT', 'AKAZE_MLDB'],
                           default='SIFT',
                           type=str.upper,
                           help="Set the feature descriptors method.")
    opts_desc.add_argument('--describer-preset',
                           choices=['NORMAL', 'HIGH', 'ULTRA'], default='HIGH',
                           type=str.upper,
                           help='Set the description detail level.')
    opts_desc.add_argument('--describer-upright', '-u',
                           action=argparse.BooleanOptionalAction,
                           help='Disable rotational invariance for feature '
                                'detection step. Useful if the camera is '
                                'always "upright" w.r.t the ground plane.')

    opts_matcher = parser.add_argument_group('matcher options')
    opts_matcher.add_argument('--matching-method',
                              choices=['AUTO',
                                       'BRUTEFORCEL2',
                                       'ANNL2',
                                       'CASCADEHASHINGL2',
                                       'FASTCASCADEHASHINGL2',
                                       'BRUTEFORCEHAMMING'],
                              default='FASTCASCADEHASHINGL2', type=str.upper,
                              help='Feature matching method.')
    opts_matcher.add_argument('--matching-geometric-model',
                              choices=['f', 'e', 'h', 'a', 'u', 'o'],
                              type=str.lower,
                              help='Geometric model for robust putative '
                                   'matches filtering: f: Fundamental, '
                                   'e: Essential, h: Homography, a: essential '
                                   'matrix with angular parameterization, u: '
                                   'upright essential matrix with angular '
                                   'parameterization, o: orthographic '
                                   'essential matrix')
    opts_matcher.add_argument('--matching-ratio', type=float, default=None,
                              help='Nearest-Neighbor distance ratio')

    opts_mvg = parser.add_argument_group('mvg reconstruction options')
    opts_mvg.add_argument('--mvg-recon-method', '-m',
                          choices=['global',
                                   'incremental',
                                   'incrementalv2',
                                   'direct'],
                          type=str.lower, default='global',
                          help='MVG scene reconstruction method. Note: direct '
                               'requires an sfm file with camera poses.')
    opts_mvg.add_argument('--mvg-priors', action=argparse.BooleanOptionalAction,
                          help='Use pose priors with SfM reconstruction')
    opts_mvg.add_argument('--mvg-refine-intrinsics', type=str.upper,
                          help='SfM intrinsic refinement options: NONE, '
                               'ADJUST_FOCAL_LENGTH, ADJUST_PRINCIPAL_POINT, '
                               'ADJUST_DISTORTION, ADJUST_ALL. '
                               'Note: Quoted options can be combined with '
                               '\'|\' (e.g. '
                               '\'ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\')')
    opts_mvg.add_argument('--mvg-robust', '-r',
                          action=argparse.BooleanOptionalAction,
                          help='robustly triangulate reconstructed scene')
    opts_mvg.add_argument('--mvg-autoscale', type=float,
                          help='use pgs-global-scaler to automatically scale '
                               'aruco markers to the provided size in '
                               'real world units. see the pgs-global-scaler '
                               '--marker-size flag for more information')
    opts_mvg.add_argument('--autoscale-method', default='markers',
                          choices=['markers', 'sample-square'],
                          help='marker detection method. see the '
                               'pgs-global-scaler --detection-method flag '
                               'for more information'
                          )
    opts_mvg.add_argument('--autoscale-marker-pix', type=int,
                          help="Minimum marker size in pixels. see the "
                               "pgs-global-scaler --min-marker-size flag "
                               "for more information")
    opts_mvg.add_argument('--autoscale-include-from',
                          help='text file containing a list of scene image '
                               'files to be exclusively considered during '
                               'auto-scaling. see the pgs-global-scaler '
                               '--include-from flag for more information')
    opts_mvg.add_argument('--autoscale-exclude-from',
                          help='text file containing a list of scene image '
                               'files to be excluded during auto-scaling. '
                               'see the pgs-global-scaler --exclude-from flag '
                               'for more information')
    opts_mvg.add_argument('--autoscale-histogram-out',
                          help='write the marker-level scale histogram to this '
                               'path')
    opts_mvg.add_argument('--autoscale-debug-dir',
                          help='write scaler marker-detection overlays to this '
                               'directory')

    # MVG hidden opts
    opts_mvg.add_argument('--cam-db', type=str, help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--sfm-ba', action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--robust-ba', action=argparse.BooleanOptionalAction,
                          help=configargparse.SUPPRESS)
    opts_mvg.add_argument('--mvg-initializer',
                          choices=['EXISTING_POSE', 'MAX_PAIR', 'AUTO_PAIR',
                                   'STELLAR'],
                          type=str.upper, help=configargparse.SUPPRESS)

    opts_mvs = parser.add_argument_group('openmvs options')
    opts_mvs.add_argument('--free-space-support',
                          action=argparse.BooleanOptionalAction,
                          help='use free-space support in ReconstructMesh')
    opts_mvs.add_argument('--mvs-densify',
                          action=argparse.BooleanOptionalAction,
                          help='Enable point cloud densification step')
    opts_mvs.add_argument('--mvs-refine', default=True,
                          action=argparse.BooleanOptionalAction,
                          help='Enable MVS mesh refinement step')
    opts_mvs.add_argument('--mvs-smooth', type=int, default=2,
                          help='Number of smoothing iterations after initial '
                               'surface reconstruction. 0 is disabled.')
    opts_mvs.add_argument('--densify-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'DensifyPointCloud')
    opts_mvs.add_argument('--refine-resolution-level', default=None, type=int,
                          help='scale input images down N times before')
    opts_mvs.add_argument('--refine-min-resolution', default=None, type=int,
                          help='do not scale images\' max dimension smaller '
                               'than this value when using '
                               '--refine-resolution-level')
    opts_mvs.add_argument('--refine-scales', default=3, type=int,
                          help='number of mesh optimization iterations on '
                               'multi-scale images')
    opts_mvs.add_argument('--refine-scale-step', default=None, type=float,
                          help='image scale factor used at each mesh '
                               'optimization step')
    opts_mvs.add_argument('--texture-resolution-level', default=None, type=int,
                          help='how many times to scale down images before '
                               'TextureMesh')
    opts_mvs.add_argument('--decimation-factor', type=float,
                          help='Decimation factor in range [0..1] to be '
                               'applied to the input surface before mesh '
                               'refinement (0 - auto, 1 - disabled)')
    opts_mvs.add_argument('--mask-value', type=int,
                          help='Label value in the image mask to ignore during '
                               'mesh densification. By default, image masks '
                               'are ignored during this step.')
    opts_mvs.add_argument('--texture-max-size', type=int, default=0,
                          help='Limits the maximum size (edge length) of the'
                               'output texture image. If set to 0 (default), '
                               'the edge length is unbounded.')
    args = parser.parse_args()

    if args.stop_after == 'autoscale' and args.mvg_autoscale is None:
        parser.error('--stop-after autoscale requires --mvg-autoscale')
    if ((args.autoscale_histogram_out or args.autoscale_debug_dir)
            and args.mvg_autoscale is None):
        parser.error('autoscale diagnostics require --mvg-autoscale')
    if args.resume_sfm:
        incompatible = []
        if args.stop_after:
            incompatible.append('--stop-after')
        if args.mvg_autoscale is not None:
            incompatible.append('--mvg-autoscale')
        if args.mvg_robust:
            incompatible.append('--mvg-robust')
        if incompatible:
            parser.error('--resume-sfm is incompatible with ' +
                         ', '.join(incompatible))

    setup_logging(args.log_level)
    logger = logging.getLogger("pgs-recon")

    # Structure for storing important paths
    logger.info('Setting up output directories')
    paths = {
        'PATH': Path(args.path).resolve(),
        'input': Path(args.input),
        'output': Path(args.output)
    }
    if args.import_calib:
        paths['input_calib'] = Path(args.import_calib)
    paths['BIN'] = paths['PATH'] / 'bin'
    paths['MVS_BIN'] = paths['BIN'] / 'OpenMVS'
    if args.cam_db is None:
        db_path = 'lib/openMVG/sensor_width_camera_database.txt'
        paths['CAM_DB'] = paths['PATH'] / db_path
    else:
        paths['CAM_DB'] = Path(args.cam_db).resolve()

    # Setup output directory names
    paths['mvg'] = paths['output'] / 'mvg'
    paths['matches_dir'] = paths['mvg'] / 'matches_dir'
    paths['matches_file'] = paths['matches_dir'] / 'matches.bin'
    paths['recon_dir'] = paths['mvg'] / 'recon_dir'
    paths['sfm'] = paths['mvg'] / 'sfm_data.json'
    paths['mvs'] = paths['output'] / 'mvs'
    paths['mvs_scene'] = paths['mvs'] / 'scene.mvs'
    paths['mvs_images'] = paths['mvs'] / 'undistorted_images'

    # Create output folders
    for d in 'output', 'mvg', 'matches_dir', 'recon_dir', 'mvs':
        paths[d].mkdir(exist_ok=True, parents=True)

    # Setup experiment
    experiment_start = dt.now(tz.utc)
    datetime_str = experiment_start.strftime('%Y%m%d%H%M%S')
    if args.name is None:
        args.name = datetime_str + '_' + str(Path(args.input).stem)
        config = paths['output'] / f'{args.name}_recon_config.txt'
    else:
        config = paths['output'] / f'{datetime_str}_{args.name}_recon_config.txt'

    # Write config after all arguments have been changed
    args.config = str(config)
    with config.open(mode='w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            arg = arg.replace('_', '-')
            file.write(f'{arg} = {attr}\n')

    # Init metadata
    metadata = {'args': " ".join(sys.argv),
                'parsed': vars(args),
                'paths': {key: str(val) for key, val in paths.items()},
                'commands': {}}
    paths['metadata'] = paths['output'] / 'metadata.json'

    # Register a metadata write whenever the program closes
    @atexit.register
    def write_metadata():
        metadata['paths'] = {key: str(val) for key, val in paths.items()}
        metadata['commands'][current_timestamp()] = "Writing metadata"
        with paths['metadata'].open('w') as meta_file:
            meta_file.write(json.dumps(metadata, indent=4, sort_keys=False))

    # Write metadata once
    write_metadata()

    if args.resume_sfm:
        resume_path = Path(args.resume_sfm).resolve()
        if not resume_path.is_file() or resume_path.stat().st_size == 0:
            parser.error(f'--resume-sfm is not a non-empty file: {resume_path}')
        paths['sfm_resume'] = resume_path
        sfm_key = 'sfm_resume'
        metadata['commands'][current_timestamp()] = (
            f'Resuming from existing SfM scene: {resume_path}')
        logger.info(f'Resuming from existing SfM scene: {resume_path}')
    else:
        logger.info('Importing dataset')
        if args.import_pgs_scan:
            init_sfm_pgs(paths, metadata=metadata)
        elif args.new_importer:
            init_sfm_generic2(paths['input'].resolve(),
                              sfm_file=paths['sfm'],
                              camdb_path=paths['CAM_DB'])
        else:
            init_sfm_generic(paths, focal_length=args.focal_length,
                             metadata=metadata)

        logger.info('Computing image features')
        compute_features(paths, method=args.describer_method,
                         preset=args.describer_preset,
                         upright=args.describer_upright,
                         metadata=metadata, threads=args.threads)

        logger.info('Matching image features')
        compute_matches(paths, method=args.matching_method,
                        ratio=args.matching_ratio, metadata=metadata)

        logger.info('Filtering image features')
        geometric_filter(paths, model=args.matching_geometric_model,
                         metadata=metadata)

        if args.mvg_recon_method == 'direct':
            logger.info('Computing structure from known poses')
            sfm_key = mvg_compute_known(paths, sfm_key='sfm', direct=True,
                                        bundle_adjustment=args.sfm_ba,
                                        metadata=metadata)
        else:
            logger.info(f'Running SfM (Engine: {args.mvg_recon_method}')
            sfm_key = mvg_sfm(paths, sfm_key='sfm',
                              engine=args.mvg_recon_method,
                              use_priors=args.mvg_priors,
                              refine_intrinsics=args.mvg_refine_intrinsics,
                              initializer=args.mvg_initializer,
                              metadata=metadata)

        # Robustification, when requested by other workflows, must precede
        # scaling so OpenMVS receives the same scene that was measured.
        if args.mvg_robust:
            logger.info('Performing robust triangulation')
            sfm_key = mvg_compute_known(paths, sfm_key=sfm_key,
                                        bundle_adjustment=args.robust_ba,
                                        metadata=metadata)

        if args.stop_after == 'sfm':
            logger.info('Colorizing SfM scene for review')
            mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)
            metadata['commands'][current_timestamp()] = 'Stopped after SfM'
            logger.info(f'Stopped after SfM: {paths[sfm_key]}')
            return

        if args.mvg_autoscale is not None:
            logger.info('Auto-scaling SfM scene')
            sfm_key = mvg_autoscale(
                paths=paths,
                sfm_key=sfm_key,
                marker_size=args.mvg_autoscale,
                detection_method=args.autoscale_method,
                marker_pix=args.autoscale_marker_pix,
                include_from=args.autoscale_include_from,
                exclude_from=args.autoscale_exclude_from,
                histogram_out=args.autoscale_histogram_out,
                debug_dir=args.autoscale_debug_dir,
                metadata=metadata)

        if args.stop_after == 'autoscale':
            logger.info('Colorizing scaled SfM scene for review')
            mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)
            metadata['commands'][current_timestamp()] = 'Stopped after autoscale'
            logger.info(f'Stopped after autoscale: {paths[sfm_key]}')
            return

    logger.info('Colorizing SfM scene')
    mvg_colorize_sfm(paths, sfm_key=sfm_key, metadata=metadata)

    # Convert MVG -> MVS
    logger.info('Converting MVG scene to MVS scene')
    mvs_key = mvg_to_mvs(paths, sfm_key=sfm_key, metadata=metadata,
                         threads=args.threads)

    # Densify MVS Scene
    if args.mvs_densify:
        logger.info('Densifying point cloud')
        mvs_key = mvs_densify(paths, mvs_key=mvs_key,
                              resolution_lvl=args.densify_resolution_level,
                              mask_value=args.mask_value,
                              metadata=metadata)

    # Reconstruct Scene Mesh
    logger.info('Reconstructing mesh')
    mvs_key, mesh_key = mvs_reconstruct(paths, mvs_key=mvs_key,
                              free_space=args.free_space_support,
                              smooth=args.mvs_smooth,
                              metadata=metadata)

    # Refine Mesh
    logger.info('Refining mesh')
    if args.mvs_refine:
        mvs_key, mesh_key = mvs_refine(paths, mvs_key=mvs_key,
                                       mesh_key=mesh_key,
                                       decimation_factor=args.decimation_factor,
                                       resolution_lvl=args.refine_resolution_level,
                                       min_resolution=args.refine_min_resolution,
                                       scales=args.refine_scales,
                                       scale_step=args.refine_scale_step,
                                       metadata=metadata)

    # Texture Mesh
    logger.info('Texturing mesh')
    mvs_texture(paths, mvs_key=mvs_key, mesh_key=mesh_key,
                file_format=args.file_type,
                resolution_lvl=args.texture_resolution_level,
                max_size=args.texture_max_size,
                metadata=metadata, output_name=args.name)

    # Add final timestamp
    current_time = current_timestamp()
    metadata['commands'][current_time] = "Processing complete"
    logger.info(f'Processing complete. Results saved to: {paths["output"]}')


if __name__ == '__main__':
    main()
