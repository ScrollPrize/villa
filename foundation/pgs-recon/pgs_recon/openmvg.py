from pathlib import Path
from typing import Dict

from pgs_recon.utility import current_timestamp, run_command


def init_sfm_generic(paths: Dict[str, Path], focal_length=None,
                     metadata: Dict = None):
    """Init sfm scene from dir of images"""
    command = [
        str(paths['BIN'] / 'openMVG_main_SfMInit_ImageListing'),
        '-i', str(paths['input'].resolve()),
        '-o', str(paths['mvg']),
        '-d', str(paths['CAM_DB']),
    ]
    if focal_length is not None:
        command.extend(['-f', str(focal_length)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)


def compute_features(paths: Dict[str, Path], method: str, preset: str,
                     upright=False, threads: int = None,
                     metadata: Dict = None):
    """MVG: Compute image features"""
    # Compute features
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeFeatures'),
        '-i', str(paths['sfm']),
        '-o', str(paths['matches_dir']),
        '-m', method,
        '-p', preset,
    ]
    if upright:
        command.extend(['-u', '1'])
    if threads is not None:
        command.extend(['-n', str(threads)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)


def compute_matches(paths: Dict[str, Path], method: str,
                    ratio: float = None,
                    metadata: Dict = None):
    """Compute image feature matches"""
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeMatches'),
        '-i', str(paths['sfm']),
        '-o', str(paths['matches_file']),
        '-n', method,
    ]
    if ratio is not None:
        command.extend(['-r', str(ratio)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)


def geometric_filter(paths: Dict[str, Path], model: str = None,
                     metadata: Dict = None):
    filtered = paths['matches_file']
    filtered = filtered.parent / (filtered.stem + '_filtered' + filtered.suffix)
    paths['matches_file_filtered'] = filtered
    command = [
        str(paths['BIN'] / 'openMVG_main_GeometricFilter'),
        '-i', str(paths['sfm']),
        '-m', str(paths['matches_file']),
        '-o', str(filtered)
    ]
    if model is not None:
        command.extend(['-g', model.lower()])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)


def mvg_sfm(paths: Dict[str, Path], sfm_key: str, engine: str, use_priors=False,
            refine_intrinsics: str = None,
            initializer: str = None,
            metadata: Dict = None) -> str:
    """Run SfM"""
    command = [
        str(paths['BIN'] / 'openMVG_main_SfM'),
        '-i', str(paths[sfm_key]),
        '-s', engine.upper(),
        '-m', str(paths['matches_dir']),
        '-o', str(paths['recon_dir']),
        '-M', str(paths['matches_file_filtered'].name),
    ]
    if use_priors:
        command.append('-P')
        if engine == 'incrementalv2':
            command.extend(['-S', 'EXISTING_POSE'])
    if refine_intrinsics is not None:
        command.extend(['-f', refine_intrinsics])
    if initializer is not None:
        command.extend(['-S', initializer])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    paths['sfm_recon'] = paths['recon_dir'] / 'sfm_data.bin'
    return 'sfm_recon'


def mvg_autoscale(paths: Dict[str, Path], sfm_key: str, marker_size: float,
                  detection_method: str = 'markers', marker_pix: int = None,
                  include_from: str = None, exclude_from: str = None,
                  histogram_out: str = None, debug_dir: str = None,
                  metadata: Dict = None) -> str:
    """Run pgs-global-scaler"""
    out_key = sfm_key + '_scaled'
    in_path = paths[sfm_key]
    paths[out_key] = paths['recon_dir'] / (in_path.stem + '_scaled.bin')
    command = [
        str(paths['BIN'] / 'pgs-global-scaler'),
        '-i', str(paths[sfm_key]),
        '-o', str(paths[out_key]),
        '-s', str(marker_size),
        '-m', detection_method,
        '--save-landmarks', str(paths['recon_dir'] / 'landmarks.ply'),
        '--save-scaled-landmarks', str(paths['recon_dir'] / 'landmarks_scaled.ply')
    ]
    if marker_pix is not None:
        command.extend(['--min-marker-pix', str(marker_pix)])
    if include_from is not None:
        command.extend(['--include-from', str(include_from)])
    if exclude_from is not None:
        command.extend(['--exclude-from', str(exclude_from)])
    if histogram_out is not None:
        command.extend(['--histogram-out', str(histogram_out)])
    if debug_dir is not None:
        command.extend(['--save-debug-images', str(debug_dir)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvg_compute_known(paths: Dict[str, Path], sfm_key: str,
                      direct: bool = False, bundle_adjustment: bool = False,
                      metadata: Dict = None) -> str:
    """Compute structure from known poses (direct/robust)"""
    out_key = sfm_key + '_structured'
    in_path = paths[sfm_key]
    paths[out_key] = paths['recon_dir'] / (in_path.stem + '_structured.bin')
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeStructureFromKnownPoses'),
        '-i', str(paths[sfm_key]),
        '-m', str(paths['matches_dir']),
        '-o', str(paths[out_key]),
        '-f', str(paths['matches_file']),
    ]
    if direct:
        command.append('-d')
    if bundle_adjustment:
        command.append('-b')
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvg_colorize_sfm(paths: Dict[str, Path], sfm_key: str,
                     metadata: Dict = None) -> str:
    """Colorize SfM file"""
    sfm_colorized_key = sfm_key + '_colorized'
    in_path = paths[sfm_key]
    paths[sfm_colorized_key] = in_path.parent / (
                in_path.stem + '_colorized.ply')
    command = [
        str(paths['BIN'] / 'openMVG_main_ComputeSfM_DataColor'),
        '-i', str(paths[sfm_key]),
        '-o', str(paths[sfm_colorized_key]),
    ]
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return sfm_colorized_key


def mvg_to_mvs(paths: Dict[str, Path], sfm_key: str, threads: int = None,
               metadata: Dict = None) -> str:
    """Convert OpenMVG SfM to OpenMVS Scene"""
    command = [
        str(paths['BIN'] / 'openMVG_main_openMVG2openMVS'),
        '-i', str(paths[sfm_key].resolve()),
        '-o', str(paths['mvs_scene'].name),
        '-d', str(paths['mvs_images'].name)
    ]
    if threads is not None:
        command.extend(['-n', str(threads)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command, cwd=paths['mvs'])
    return 'mvs_scene'
