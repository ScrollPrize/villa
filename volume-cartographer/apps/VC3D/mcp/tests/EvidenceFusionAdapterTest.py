#!/usr/bin/env python3
"""Synthetic Phase 4 stability and evidence-ranking regression tests."""
from __future__ import annotations

import argparse, importlib.util, json, tempfile
from pathlib import Path
import numpy as np
import zarr


def load(path: Path):
    spec=importlib.util.spec_from_file_location("evidence_fusion_adapter",path);assert spec and spec.loader
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def registered(path:Path,shift:float=0,signal_shift:float=0)->Path:
    group=zarr.open_group(str(path/'surface.zarr'),mode='w',zarr_format=2);geometry=group.require_group('geometry');renders=group.require_group('renders')
    v,u=np.mgrid[:64,:64];xyz=np.stack((u.astype(float)+shift,v,np.full_like(u,30)),axis=-1).astype(np.float32);signal=(u+v).astype(np.float32)+signal_shift
    geometry.create_array('xyz',data=xyz,chunks=(32,32,3));geometry.create_array('valid',data=np.ones((64,64),np.uint8),chunks=(32,32));renders.create_array('raw',data=signal,chunks=(32,32))
    group.attrs.update({'coordinate_space':'ct_l0_xyz','registered_volume':{'voxel_spacing':[100,100,100],'voxel_spacing_unit':'um','voxel_spacing_explicit':True}})
    (path/'manifest.json').write_text('{}');return path


def artifact(path:Path,manifest:dict)->str:
    path.mkdir(parents=True);(path/'manifest.json').write_text(json.dumps(manifest));return str(path)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',required=True,type=Path);args=parser.parse_args();adapter=load(args.adapter)
    with tempfile.TemporaryDirectory(prefix='vc-evidence-fusion-') as temporary:
        root=Path(temporary);base=registered(root/'base');v1=registered(root/'v1',.02,.1);v2=registered(root/'v2',.05,.2)
        request={'baseline':{'job_id':'base','artifact_id':'registered-surface'},'variants':[{'job_id':'v1','artifact_id':'registered-surface'},{'job_id':'v2','artifact_id':'registered-surface'}],'displacement_scale_mm':.05,'normal_angle_scale_degrees':10,'signal_scale':.1}
        out=root/'stability';result=adapter.perturbation_stability(request,base,[v1,v2],out)
        assert 0<=result['overall_stability_score']<=1;assert len(result['comparisons'])==2;assert (out/'surface-stability.zarr'/'.zgroup').is_file()
        source={'job_id':'base','artifact_id':'registered-surface'};shape=[64,64]

        vv,uu=np.mgrid[:64,:64];ink_model_raw=((uu+vv)/90-.2).astype(np.float32);dinovol_raw=(np.sin(uu/8)+np.cos(vv/9)).astype(np.float32)
        ink_model=root/'ink-model';ink_model.mkdir();np.save(ink_model/'ink-model-score.npy',ink_model_raw);ink_model_support=np.ones((64,64),np.uint8);ink_model_support[0]=0;np.save(ink_model/'ink-valid.npy',ink_model_support)
        (ink_model/'manifest.json').write_text(json.dumps({'kind':'resnet152_ink_model_score_v1','output_shape_hw':shape,'source_surface_artifact':source}))
        dinovol=root/'dinovol';dinovol.mkdir();np.save(dinovol/'exemplar-similarity-surface.npy',dinovol_raw);dinovol_support=np.ones((64,64),np.uint8);dinovol_support[:,0]=0;np.save(dinovol/'surface-support.npy',dinovol_support)
        (dinovol/'manifest.json').write_text(json.dumps({'kind':'dinovol_registered_exemplar_v1','surface_shape_vu':shape,'source_surface_artifact':source}))
        fusion_request={'ink_model':{'job_id':'ink-model','artifact_id':'ink-prediction'},'ink_model_path':str(ink_model),'dinovol':{'job_id':'dinovol','artifact_id':'dinovol-exemplar'},'dinovol_path':str(dinovol),'stability':{'job_id':'stability','artifact_id':'surface-stability'},'stability_path':str(out),'weights':{'ink_model':2,'dinovol':1,'stability':1}}
        fusion=adapter.fuse_ink_scores(fusion_request,root/'fusion');fused=zarr.open_group(str(root/'fusion/ink-fusion.zarr'),mode='r')
        expected_arrays={'ink_model_score_raw','ink_model_score','dinovol_similarity_raw','dinovol_similarity_normalized','stability_local_score_raw','stability_local_score','combined_score','valid'}
        assert set(fused.array_keys())==expected_arrays;assert np.array_equal(fused['ink_model_score_raw'][:],ink_model_raw)
        assert np.array_equal(fused['dinovol_similarity_raw'][:],dinovol_raw);assert fusion['source_surface_consistency']=='verified'
        assert fusion['valid_pixels']==63*63;assert (root/'fusion/fusion-valid.npy').is_file();assert set(fusion['component_maps_preserved'])==expected_arrays-{'combined_score','valid'}
        y,x=10,10;low=fusion['dinovol_normalization']['low'];high=fusion['dinovol_normalization']['high'];dn=np.clip((dinovol_raw[y,x]-low)/(high-low),0,1);stability=np.load(out/'stability-local-score.npy')[y,x]
        assert np.isclose(fused['combined_score'][y,x],(2*np.clip(ink_model_raw[y,x],0,1)+dn+stability)/4)
        try: adapter.fuse_ink_scores({**fusion_request,'weights':{'mystery':1}},root/'bad-weight')
        except ValueError as error: assert 'unknown fusion weight' in str(error)
        else: raise AssertionError('unknown fusion weight accepted')
        without_stability={key:value for key,value in fusion_request.items() if key not in {'stability','stability_path'}};without_stability['weights']={'stability':1}
        try: adapter.fuse_ink_scores(without_stability,root/'bad-stability')
        except ValueError as error: assert 'requires a stability artifact' in str(error)
        else: raise AssertionError('stability weight accepted without stability artifact')

        geometry=artifact(root/'geometry',{'source_artifact':source,'surface_shape_vu':shape,'valid_pixels':4096,'p95_stretch_log_ratio':.02,'p95_normal_change_degrees':2,'fold_or_degenerate_cells':0,'connected_components':1,'enclosed_holes':0})
        alignment=artifact(root/'alignment',{'source_artifact':source,'surface_shape_vu':shape,'support_fraction':.98,'median_confidence':.8,'median_peak_offset_voxels':.2})
        grid=artifact(root/'grid',{'source_artifact':source,'surface_shape_vu':shape,'pixel_spacing_mm_uv':[.1,.1],'score_summary':{'median':1.2},'significance':{'empirical_p_value':.05}})
        resolved=[{'id':'candidate-a','resolved':{'geometry':geometry,'alignment':alignment,'grid':grid,'stability':str(out)}},{'id':'candidate-b','resolved':{'geometry':geometry,'alignment':alignment,'grid':grid}}]
        ranking=adapter.rank_evidence({'resolved_candidates':resolved,'weights':{'stability':2}},root/'ranking')
        assert len(ranking['ranked_candidates'])==2;assert ranking['ranked_candidates'][0]['combined_score']<=1;assert (root/'ranking'/'evidence-ranking.png').is_file()
    print('EvidenceFusionAdapterTest passed');return 0

if __name__=='__main__':raise SystemExit(main())
