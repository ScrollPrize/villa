#!/usr/bin/env python3
"""Adversarial and malformed-input boundaries for the evidence pipeline."""
from __future__ import annotations
import argparse,importlib.util,json,tempfile
from pathlib import Path
import numpy as np,zarr

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);assert spec and spec.loader;m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def rejects(fn,fragment):
 try:fn()
 except Exception as error:
  assert fragment.lower() in str(error).lower(),(fragment,error);return
 raise AssertionError(f'expected rejection containing {fragment}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--mcp-dir',required=True,type=Path);a=p.parse_args();d=a.mcp_dir
 stager=module('volume_stager_boundary',d/'volume_stager.py');bundle=module('surface_bundle_boundary',d/'surface_bundle_adapter.py');structural=module('structural_boundary',d/'structural_evidence_adapter.py');fusion=module('fusion_boundary',d/'evidence_fusion_adapter.py');review=module('review_boundary',d/'review_adapter.py')
 assert not stager.allowed_remote_uri('https://dl.ash2txt.org.evil.invalid/a.zarr');assert not stager.allowed_remote_uri('https://dl.ash2txt.org/a/%2e%2e/private')
 with tempfile.TemporaryDirectory(prefix='vc-boundaries-') as temporary:
  root=Path(temporary);rejects(lambda:bundle.read_tifxyz(root/'missing'),'tifxyz directory')
  artifact=root/'registered';group=zarr.open_group(str(artifact/'surface.zarr'),mode='w',zarr_format=2);geo=group.require_group('geometry');renders=group.require_group('renders');v,u=np.mgrid[:16,:16];xyz=np.stack((u,v,np.ones_like(u)*3),axis=-1).astype(np.float32);geo.create_array('xyz',data=xyz,chunks=(8,8,3));geo.create_array('valid',data=np.ones((16,16),np.uint8),chunks=(8,8));renders.create_array('raw',data=(u+v).astype(np.float32),chunks=(8,8));group.attrs['registered_volume']={'voxel_spacing':[100,100,100],'voxel_spacing_unit':'um','voxel_spacing_explicit':False};(artifact/'manifest.json').write_text('{}')
  rejects(lambda:structural.grid_coherence({'surface':{},'letter_period_mm':1,'null_trials':4},artifact,root/'grid'),'explicit physical')
  group.attrs['registered_volume']={'voxel_spacing':[100,100,100],'voxel_spacing_unit':'um','voxel_spacing_explicit':True}
  rejects(lambda:structural.grid_coherence({'surface':{},'letter_period_mm':1,'null_trials':2},artifact,root/'grid'),'null_trials')
  queue=root/'queue';queue.mkdir();document={'kind':'vc_review_queue_v1','queue_digest':'abc','items':[{'item_id':'candidate:a','kind':'candidate','priority':.5}]};(queue/'review-queue.json').write_text(json.dumps(document))
  rejects(lambda:review.record_assessment({'queue':{},'reviewer_id':'bad reviewer','assessments':[{'item_id':'candidate:a','decision':'accept'}]},queue,root/'assess'),'reviewer_id')
  rejects(lambda:review.record_assessment({'queue':{},'reviewer_id':'good','assessments':[{'item_id':'missing','decision':'accept'}]},queue,root/'assess'),'not found')
  a1=root/'a1';a2=root/'a2';a1.mkdir();a2.mkdir();(a1/'review-assessment.json').write_text(json.dumps({'queue_digest':'one','reviewer_id':'a','records':[]}));(a2/'review-assessment.json').write_text(json.dumps({'queue_digest':'two','reviewer_id':'b','records':[]}));rejects(lambda:review.evaluate({'assessments':[]},[a1,a2],root/'evaluation'),'same immutable')
  rejects(lambda:fusion.rank_evidence({'resolved_candidates':[]},root/'rank'),'1 to 16')
 print('McpEvidenceBoundaryTest passed');return 0
if __name__=='__main__':raise SystemExit(main())
