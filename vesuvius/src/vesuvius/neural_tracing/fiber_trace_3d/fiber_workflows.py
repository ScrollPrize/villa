"""Shareable fiber-layout preparation, search, evaluation and export workflows.

Run ``python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows --config FILE``.
Configuration selects exactly one named workflow or a list of native commands.
GPU stages retain their existing manager-based launchers.
"""
import argparse
import json
from pathlib import Path

def run(config):
    if 'commands' not in config and len(config)!=1:
        raise ValueError('Select exactly one named workflow')
    if 'fiber_pair_connectivity' in config:
        from .fiber_pair_connectivity import run_pair_connectivity
        return run_pair_connectivity(config['fiber_pair_connectivity'])
    if 'fiber_split_weight_search_control' in config:
        from .fiber_split_weight_search import hold_existing
        return hold_existing(config['fiber_split_weight_search_control'])
    if 'fiber_split_weight_search' in config:
        from .fiber_split_weight_search import run as split_search
        return split_search(config['fiber_split_weight_search'])
    if 'fiber_neighbor_pair' in config:
        from .fiber_neighbor_pair import run as run_neighbor_pair
        return run_neighbor_pair(config['fiber_neighbor_pair'])
    if 'fiber_neighbor_expansion' in config:
        from .fiber_neighbor_expansion import run as expand_neighbors
        return expand_neighbors(config['fiber_neighbor_expansion'])
    if 'paired_winding_review' in config:
        from .paired_winding_review import run as paired_review
        return paired_review(config['paired_winding_review'])
    if 'fiber_paired_weight_search' in config:
        from .fiber_paired_weight_search import run as paired_search
        return paired_search(config['fiber_paired_weight_search'])
    if 'fiber_solver_trial' in config:
        settings=config['fiber_solver_trial']
        recipe=json.loads(Path(settings['recipe']).read_text())
        if len(recipe)!=1:raise ValueError('Trial needs one model recipe')
        key=next(iter(recipe));trial=recipe[key]
        if key not in ('fiber_selected_dataset_eval','fiber_gt_convergence_probe'):
            raise ValueError('Unsupported solver trial recipe')
        trial['output_root']=settings['output_root']
        trial.setdefault('parameters',{}).update(settings.get('parameters',{}))
        if 'devices' in settings:trial['devices']=settings['devices']
        from . import fiber_workflows
        return fiber_workflows.run(recipe)
    if 'fiber_reference_check' in config:
        from .fiber_reference_check import run
        return run(config['fiber_reference_check'])
    if 'fiber_optimizer_probe' in config:
        from .fiber_optimizer_probe import run
        return run(config['fiber_optimizer_probe'])
    if 'fiber_gt_width_initialization' in config:
        from .fiber_gt_width_initialization import run
        return run(config['fiber_gt_width_initialization'])
    if 'fiber_gt_convergence_probe' in config:
        from .fiber_gt_convergence_probe import run
        return run(config['fiber_gt_convergence_probe'])
    if 'fiber_fixed_metric_probe' in config:
        from .fiber_fixed_metric_probe import run
        return run(config['fiber_fixed_metric_probe'])
    if 'fiber_current_gt_solve' in config:
        from .fiber_current_gt_solve import run
        return run(config['fiber_current_gt_solve'])
    if 'fiber_bucket_uv_report' in config:
        from .fiber_bucket_uv import report
        return report(config['fiber_bucket_uv_report'])
    if 'fiber_lbfgs_audit' in config:
        from .fiber_lbfgs_audit import run
        return run(config['fiber_lbfgs_audit'])
    if 'fiber_selected_score' in config:
        from .fiber_selected_dataset_eval import score_completed
        return score_completed(config['fiber_selected_score'])
    if 'fiber_forced_selectable_audit' in config:
        from .fiber_forced_selectable_audit import run
        return run(config['fiber_forced_selectable_audit'])
    if 'fiber_selected_checkpoint_report' in config:
        from .fiber_selected_dataset_eval import checkpoint_scores
        return checkpoint_scores(config['fiber_selected_checkpoint_report'])
    if 'fiber_selected_dataset_eval' in config:
        from .fiber_selected_dataset_eval import run
        return run(config['fiber_selected_dataset_eval'])
    if 'fiber_full_slab_soft' in config:
        from .fiber_full_slab import run_soft_trial
        return run_soft_trial(config['fiber_full_slab_soft'])
    if 'fiber_full_slab_progress' in config:
        from .fiber_full_slab import run_progress
        return run_progress(config['fiber_full_slab_progress'])
    if 'fiber_full_slab' in config:
        from .fiber_full_slab import run
        return run(config['fiber_full_slab'])
    if 'fiber_gt_inpainting' in config:
        from .fiber_gt_inpainting import run
        return run(config['fiber_gt_inpainting'])
    if 'fiber_anchored_dataset' in config:
        from .fiber_anchored_dataset import run
        return run(config['fiber_anchored_dataset'])
    if 'fiber_density_weight_search' in config:
        from .fiber_uv_weight_search import run_density
        return run_density(config['fiber_density_weight_search'])
    if 'fiber_selected_weight_search' in config:
        from .fiber_uv_weight_search import run_selected
        return run_selected(config['fiber_selected_weight_search'])
    if 'fiber_uv_refinement_inspection' in config:
        from .fiber_uv_refinement_inspection import run as inspect_refinement
        return inspect_refinement(config['fiber_uv_refinement_inspection'])
    if 'fiber_uv_surface_refinement' in config:
        from .fiber_uv_surface_refinement import run as refine_surface
        return refine_surface(config['fiber_uv_surface_refinement'])
    if 'fiber_uv_weight_search_stop' in config:
        from .fiber_uv_weight_search import stop
        return stop(config['fiber_uv_weight_search_stop'])
    if 'fiber_reference_rounding_audit' in config:
        from .fiber_reference_rounding_audit import run as audit_reference
        return audit_reference(config['fiber_reference_rounding_audit'])
    if 'fiber_gt_constraint_score' in config:
        from .fiber_gt_constraint_score import run as score_gt
        return score_gt(config['fiber_gt_constraint_score'])
    if 'fiber_uv_inlier_metrics' in config:
        from .fiber_uv_inlier_metrics import run as score_inliers
        return score_inliers(config['fiber_uv_inlier_metrics'])
    if 'fiber_json_solve' in config:
        from .fiber_json_solve import run as json_solve
        return json_solve(config['fiber_json_solve'])
    if 'fiber_uv_reference_subset' in config:
        from .fiber_uv_reference_subset import run as reference_subset
        return reference_subset(config['fiber_uv_reference_subset'])
    if 'fiber_reference_comparison' in config:
        from .fiber_reference_comparison import run as reference_comparison
        return reference_comparison(config['fiber_reference_comparison'])
    if 'fiber_reference_surface' in config:
        from .fiber_reference_surface import run as reference_surface
        return reference_surface(config['fiber_reference_surface'])
    if 'fiber_uv_density' in config:
        from .fiber_uv_density import run as density_maps
        return density_maps(config['fiber_uv_density'])
    if 'fiber_uv_hv_rejection' in config:
        from .fiber_uv_hv_rejection import run as hv_rejection
        return hv_rejection(config['fiber_uv_hv_rejection'])
    if 'fiber_uv_kollesis' in config:
        from .fiber_uv_kollesis import run as kollesis_maps
        return kollesis_maps(config['fiber_uv_kollesis'])
    if 'fiber_uv_surface_audit' in config:
        from .fiber_uv_surface import audit as surface_audit
        return surface_audit(config['fiber_uv_surface_audit'])
    if 'fiber_uv_surface' in config:
        from .fiber_uv_surface import run as surface_uv
        return surface_uv(config['fiber_uv_surface'])
    if 'fiber_hv_report' in config:
        from .fiber_hv_stage import report
        return report(config['fiber_hv_report'])
    if 'fiber_axial_features' in config:
        from .fiber_hv_stage import run_features
        return run_features(config['fiber_axial_features'])
    if 'fiber_direction' in config or 'fiber_h_direction' in config:
        from .fiber_direction import run as run_direction
        return run_direction(config.get('fiber_direction',config.get('fiber_h_direction')))
    if 'fiber_h_direction_report' in config:
        from .fiber_direction import report as report_direction
        return report_direction(config['fiber_h_direction_report'])
    if 'fiber_direction_review' in config or 'fiber_h_direction_review' in config:
        from .fiber_direction_review import run as review_direction
        return review_direction(config.get('fiber_direction_review',config.get('fiber_h_direction_review')))
    if 'fiber_soft_winding_report' in config:
        from .fiber_soft_winding_report import run as run_soft_report
        return run_soft_report(config['fiber_soft_winding_report'])
    if 'fiber_density_sign_report' in config:
        from .fiber_density_sign_report import run as run_sign_report
        return run_sign_report(config['fiber_density_sign_report'])
    if 'fiber_implicit_match_report' in config:
        from .fiber_implicit_match_report import run as run_implicit_report
        return run_implicit_report(config['fiber_implicit_match_report'])
    if 'fiber_lbfgs_restart' in config:
        from .fiber_lbfgs_audit import restart
        return restart(config['fiber_lbfgs_restart'])
    if 'fiber_implicit_winding' in config:
        from .fiber_implicit_winding import run as run_implicit
        return run_implicit(config['fiber_implicit_winding'])
    if 'fiber_joint_gt_eval' in config:
        from .fiber_joint_gt_eval import run
        return run(config['fiber_joint_gt_eval'])
    if 'fiber_exact_deformation' in config:
        from .fiber_v_endpoint_offsets import run_exact
        return run_exact(config['fiber_exact_deformation'])
    if 'fiber_soft_winding' in config:
        from .fiber_soft_winding import run as run_soft
        return run_soft(config['fiber_soft_winding'])
    if 'fiber_uv_layout' in config:
        from .fiber_uv_layout import run as run_uv
        return run_uv(config['fiber_uv_layout'])
    if 'fiber_uv_report' in config:
        from .fiber_uv_layout import report as report_uv
        return report_uv(config['fiber_uv_report'])
    if 'fiber_uv_visualization' in config:
        from .fiber_uv_visualization import run as visualize_uv
        return visualize_uv(config['fiber_uv_visualization'])
    if 'fiber_hv_stage' in config:
        from .fiber_hv_stage import run as run_hv
        return run_hv(config['fiber_hv_stage'])
    if 'fiber_normal_arrow_update' in config:
        from .fiber_normal_sign_visualization import update_arrows
        return update_arrows(config['fiber_normal_arrow_update'])
    if 'fiber_normal_obj_repair' in config:
        from .fiber_normal_sign_visualization import repair_obj_lines
        return repair_obj_lines(config['fiber_normal_obj_repair'])
    if 'fiber_normal_visualization' in config:
        from .fiber_normal_sign_visualization import export
        return export(config['fiber_normal_visualization'])
    if 'fiber_normal_priors' in config:
        from .fiber_normal_sign import run_priors
        return run_priors(config['fiber_normal_priors'])
    if 'fiber_normal_sign' in config:
        from .fiber_normal_sign import run as run_sign
        return run_sign(config['fiber_normal_sign'])

    if 'commands' in config:
        from .fiber_native_workflow import run
        return run(config)
    raise ValueError('Unknown fiber workflow: '+', '.join(config))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if 'commands' not in config and len(config)!=1:
        parser.error('Select exactly one named workflow')
    run(config)


if __name__=='__main__':
    main()
