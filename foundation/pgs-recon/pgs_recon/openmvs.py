from pathlib import Path
from typing import Dict, Tuple

from pgs_recon.utility import current_timestamp, run_command


def mvs_densify(paths: Dict[str, Path], mvs_key: str,
                resolution_lvl: int = None, mask_value: int = None,
                metadata: Dict = None) -> str:
    """Densify a point cloud"""
    out_key = mvs_key + '_dense'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_dense.mvs')
    command = [
        str(paths['MVS_BIN'] / 'DensifyPointCloud'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs']),
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if mask_value is not None:
        command.extend(['--ignore-mask-label', str(mask_value)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvs_reconstruct(paths: Dict[str, Path], mvs_key: str, free_space=False,
                    smooth: int = 2, metadata: Dict = None) -> Tuple[str, str]:
    """Reconstruct an MVS scene"""
    mesh_key = mvs_key + '_mesh'
    scene_key = mvs_key
    in_path = paths[mvs_key]
    paths[mesh_key] = in_path.parent / (in_path.stem + '_mesh.ply')
    command = [
        str(paths['MVS_BIN'] / 'ReconstructMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[mesh_key].name),
        '-w', str(paths['mvs']),
        '--smooth', str(smooth),
    ]
    if free_space:
        command.extend(['--free-space-support', '1'])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return scene_key, mesh_key


def mvs_refine(paths: Dict[str, Path], mvs_key: str, mesh_key: str,
               decimation_factor: float = None, resolution_lvl: int = None,
               min_resolution: int = None, scales: int = 3,
               scale_step: float = None,
               metadata: Dict = None) -> Tuple[str, str]:
    """Refine a reconstructed mesh"""
    out_key = mvs_key + '_refine'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_refine.ply')
    command = [
        str(paths['MVS_BIN'] / 'RefineMesh'),
        '-i', str(paths[mvs_key].name),
        '-m', str(paths[mesh_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs'])
    ]
    if decimation_factor is not None:
        command.extend(['--decimate', str(decimation_factor)])
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if min_resolution is not None:
        command.extend(['--min-resolution', str(min_resolution)])
    if scales is not None:
        command.extend(['--scales', str(scales)])
    if scale_step is not None:
        command.extend(['--scale-step', str(scale_step)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return mvs_key, out_key


def mvs_texture(paths: Dict[str, Path], mvs_key: str, mesh_key: str,
                file_format: str = 'ply', resolution_lvl: int = None,
                max_size: int = 0, metadata: Dict = None,
                output_name: str = None) -> str:
    """Texture a mesh"""
    out_key = mvs_key + '_texture'
    in_path = paths[mvs_key]
    if output_name is not None:
        paths[out_key] = in_path.parent / f'{output_name}.{file_format.lower()}'
    else:
        paths[out_key] = in_path.parent / (
                    in_path.stem + f'_texture.{file_format.lower()}')
    command = [
        str(paths['MVS_BIN'] / 'TextureMesh'),
        '-i', str(paths[mvs_key].name),
        '-m', str(paths[mesh_key].name),
        '-o', str(paths[out_key].name),
        '--export-type', file_format.lower(),
        '-w', str(paths['mvs']),
        '--max-texture-size', str(max_size)
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key
