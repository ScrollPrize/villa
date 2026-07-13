#!/usr/bin/env python3
"""Deterministic full local pipeline fixture from TIFXYZ/Zarr through review."""
from __future__ import annotations
import argparse, importlib.util, json, tempfile
from pathlib import Path
import numpy as np, tifffile, zarr


def module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path);assert spec and spec.loader;m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tifxyz(path:Path,shift:float=0)->Path:
    path.mkdir();v,u=np.mgrid[:96,:96];tifffile.imwrite(path/'x.tif',(u+10+shift).astype(np.float32));tifffile.imwrite(path/'y.tif',(v+10).astype(np.float32));tifffile.imwrite(path/'z.tif',np.full((96,96),8,np.float32));tifffile.imwrite(path/'mask.tif',np.ones((96,96),np.uint8));(path/'meta.json').write_text(json.dumps({'format':'tifxyz','scale':[1,1]}));return path


def volume(path:Path,altered=False)->Path:
    root=zarr.open_group(str(path),mode='w',zarr_format=2);v=np.zeros((20,128,128),np.uint16);z,y,x=np.mgrid[:20,:128,:128];signal=24000+5000*np.sin(2*np.pi*x/12)+3500*np.sin(2*np.pi*y/20)
    if altered:signal+=1800*np.sin(2*np.pi*x/7)*(x>64)
    root.create_array('0',data=np.clip(signal,0,65535).astype(np.uint16),chunks=(20,64,64));return path


def registered(surface,stager,bundler,source,out,reference):
    request={'surface':reference,'coordinate_space':'ct_l0_xyz','normal_padding_voxels':2};imported=bundler.import_surface(request,surface,out);region={**imported['required_volume_region_xyz'],'space':'ct_l0_xyz'}
    stage=stager.stage({'source':{'kind':'local_zarr','path':str(source),'array_path':'0','scale':0,'voxel_spacing':[100,100,100],'voxel_spacing_unit':'um','origin_xyz':[0,0,0]},'region':region},out/'staging')
    bundler.render_surface({'surface_bundle':imported['surface_bundle'],'staged_volume':stage['staged_path'],'staged_region_xyz':stage['submitted_region_xyz'],'volume_source':stage['source'],'array_path':'0','scale':0,'voxel_spacing':stage['voxel_spacing'],'voxel_spacing_unit':'um','voxel_spacing_explicit':True,'origin_xyz':stage['origin_xyz']},out);return out


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--mcp-dir',required=True,type=Path);a=p.parse_args();d=a.mcp_dir
    stager=module('volume_stager',d/'volume_stager.py');bundler=module('surface_bundle_adapter',d/'surface_bundle_adapter.py');structural=module('structural_evidence_adapter',d/'structural_evidence_adapter.py');fusion=module('evidence_fusion_adapter',d/'evidence_fusion_adapter.py');review=module('review_adapter',d/'review_adapter.py')
    with tempfile.TemporaryDirectory(prefix='vc-e2e-evidence-') as temporary:
        root=Path(temporary);surface=tifxyz(root/'surface');shifted=tifxyz(root/'surface-shifted',.08);vol_a=volume(root/'a.zarr');vol_b=volume(root/'b.zarr',True)
        ref_a={'job_id':'registered-a','artifact_id':'registered-surface'};ref_b={'job_id':'registered-b','artifact_id':'registered-surface'};ref_shift={'job_id':'registered-shift','artifact_id':'registered-surface'}
        reg_a=registered(surface,stager,bundler,vol_a,root/'registered-a',ref_a);reg_b=registered(surface,stager,bundler,vol_b,root/'registered-b',ref_b);reg_shift=registered(shifted,stager,bundler,vol_a,root/'registered-shift',ref_shift)
        geo_a=bundler.geometry_diagnostics({'surface':ref_a},reg_a,root/'geometry-a');align_a=bundler.ct_alignment({'surface':ref_a,'maximum_offset_voxels':2},reg_a,root/'alignment-a')
        geo_b=bundler.geometry_diagnostics({'surface':ref_b},reg_b,root/'geometry-b');align_b=bundler.ct_alignment({'surface':ref_b,'maximum_offset_voxels':2},reg_b,root/'alignment-b')
        grid_args={'polarity':'bright','letter_period_mm':1.2,'line_period_mm':2.0,'window_width_mm':6,'window_height_mm':6,'step_mm':2,'minimum_cycles':3,'null_trials':4,'null_seed':3}
        grid_a=structural.grid_coherence({**grid_args,'surface':ref_a},reg_a,root/'grid-a');grid_b=structural.grid_coherence({**grid_args,'surface':ref_b},reg_b,root/'grid-b')
        comparison=structural.compare_registered({**grid_args,'surface_a':ref_a,'surface_b':ref_b},reg_a,reg_b,root/'comparison');fold=structural.epoch_fold({'surface':ref_a,'grid':{'job_id':'grid-a','artifact_id':'grid-coherence'},'polarity':'bright','period_tolerance':.1,'period_steps':21,'phase_bins':32,'null_trials':4,'null_seed':4},reg_a,root/'grid-a',root/'fold')
        stability=fusion.perturbation_stability({'baseline':ref_a,'variants':[ref_shift],'displacement_scale_mm':.05,'normal_angle_scale_degrees':10,'signal_scale':.1},reg_a,[reg_shift],root/'stability')
        candidates=[{'id':'a','resolved':{'geometry':str(root/'geometry-a'),'alignment':str(root/'alignment-a'),'grid':str(root/'grid-a'),'stability':str(root/'stability')}},{'id':'b','resolved':{'geometry':str(root/'geometry-b'),'alignment':str(root/'alignment-b'),'grid':str(root/'grid-b')}}]
        ranking=fusion.rank_evidence({'resolved_candidates':candidates,'weights':{'geometry':1,'alignment':1,'grid':1,'stability':1}},root/'ranking')
        queue=review.create_queue({'ranking':{'job_id':'ranking','artifact_id':'evidence-ranking'},'comparison':{'job_id':'comparison','artifact_id':'structural-comparison'},'max_items':20,'divergence_percentile':85},root/'ranking',root/'comparison',root/'queue')
        candidate_ids=[item['item_id'] for item in queue['items'] if item['kind']=='candidate'];assert candidate_ids
        qref={'job_id':'queue','artifact_id':'review-queue'}
        assessment_a=review.record_assessment({'queue':qref,'reviewer_id':'reviewer-a','assessments':[{'item_id':candidate_ids[0],'decision':'accept','confidence':.9}]},root/'queue',root/'assessment-a')
        assessment_b=review.record_assessment({'queue':qref,'reviewer_id':'reviewer-b','assessments':[{'item_id':candidate_ids[0],'decision':'accept','confidence':.8}]},root/'queue',root/'assessment-b')
        evaluation=review.evaluate({'assessments':[{'job_id':'assessment-a','artifact_id':'review-assessment'},{'job_id':'assessment-b','artifact_id':'review-assessment'}]},[root/'assessment-a',root/'assessment-b'],root/'evaluation')
        assert geo_a['connected_components']==1 and align_a['support_fraction']>0.99;assert grid_a['score_summary']['maximum']>0;assert comparison['maximum_xyz_difference']==0;assert fold['look_elsewhere_corrected_empirical_p_value']>0;assert 0<=stability['overall_stability_score']<=1;assert len(ranking['ranked_candidates'])==2;assert queue['queue_digest'];assert assessment_a['queue_digest']==assessment_b['queue_digest'];assert evaluation['overall_pair_agreement']==1.0
        required=[root/'registered-a/surface.zarr/.zgroup',root/'geometry-a/geometry-diagnostics.zarr/.zgroup',root/'alignment-a/ct-alignment.zarr/.zgroup',root/'grid-a/grid-coherence.zarr/.zgroup',root/'comparison/structural-comparison.zarr/.zgroup',root/'stability/surface-stability.zarr/.zgroup',root/'ranking/evidence-ranking.csv',root/'queue/review-queue.json',root/'evaluation/evaluation.json'];assert all(path.exists() for path in required)
    print('EndToEndEvidencePipelineTest passed');return 0
if __name__=='__main__':raise SystemExit(main())
