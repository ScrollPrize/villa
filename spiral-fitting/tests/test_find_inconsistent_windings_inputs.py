import fit_spiral as fs
from find_inconsistent_windings import build_fit_inputs


def test_explicit_diagnostic_inputs_override_checkpoint_source_ablation(tmp_path):
    patches_dir = tmp_path / 'patches'
    absolute_path = tmp_path / 'abs_winding.json'
    relative_path = tmp_path / 'diagnostic_links.json'
    fibers_path = tmp_path / 'fibers'
    checkpoint = {
        'cfg': {
            'input_disable_patches': True,
            'input_use_verified_patches': False,
            'input_use_unverified_patches': False,
            'input_use_tracks': False,
            'input_use_fibers': False,
            'input_use_pcl_absolute': False,
            'input_use_pcl_relative': False,
            'input_use_pcl_same_winding': False,
            'input_use_pcl_drawn_control_points': False,
        },
        'z_begin': 100,
        'z_end': 200,
        'spiral_outward_sense': 'CW',
    }

    fit_config, scroll, paths, model_z_begin, model_z_end = build_fit_inputs(
        checkpoint,
        str(patches_dir),
        (str(absolute_path), str(relative_path)),
        110,
        190,
        str(tmp_path / 'umbilicus.json'),
        str(fibers_path),
    )
    context = fs.FitContext(fit_config, scroll=scroll, paths=paths)

    assert fit_config['input_disable_patches'] is False
    assert fit_config['input_use_verified_patches'] is True
    assert fit_config['input_use_pcl_absolute'] is True
    assert fit_config['input_use_pcl_relative'] is True
    assert fit_config['input_use_fibers'] is True
    assert fit_config['input_use_unverified_patches'] is False
    assert fit_config['input_use_tracks'] is False
    assert fit_config['input_use_pcl_same_winding'] is False
    assert fit_config['input_use_pcl_drawn_control_points'] is False

    assert context.verified_patches_path == str(patches_dir)
    assert context.fibers_path == str(fibers_path)
    assert context.pcl_input_specs == [
        (str(absolute_path), None),
        (str(relative_path), None),
    ]
    assert context.unverified_patches_path is None
    assert context.tracks_dbm_path is None
    assert (model_z_begin, model_z_end) == (100, 200)
