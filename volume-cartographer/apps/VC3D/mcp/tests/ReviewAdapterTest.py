#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,tempfile
from pathlib import Path

def load(path):
 spec=importlib.util.spec_from_file_location('review_adapter',path);assert spec and spec.loader;m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def artifact(path,manifest,filename='manifest.json'):
 path.mkdir(parents=True,exist_ok=True);(path/filename).write_text(json.dumps(manifest));return path

def main():
 p=argparse.ArgumentParser();p.add_argument('--adapter',required=True,type=Path);a=p.parse_args();m=load(a.adapter)
 with tempfile.TemporaryDirectory(prefix='vc-review-') as t:
  root=Path(t);ranking=artifact(root/'ranking',{'ranked_candidates':[{'id':'a','rank':1,'combined_score':.8,'components':{'geometry':.9,'alignment':.7,'grid':.8}},{'id':'b','rank':2,'combined_score':.4,'components':{'geometry':.5,'alignment':.3,'grid':.4}}]})
  queue_request={'ranking':{'job_id':'rank','artifact_id':'evidence-ranking'},'max_items':10};queue=m.create_queue(queue_request,ranking,None,root/'queue');assert len(queue['items'])==2
  assessment_request={'queue':{'job_id':'queue','artifact_id':'review-queue'},'reviewer_id':'reviewer-a','assessments':[{'item_id':'candidate:a','decision':'accept','confidence':.9,'reason_codes':['clear']},{'item_id':'candidate:b','decision':'reject','confidence':.8}]};assessment=m.record_assessment(assessment_request,root/'queue',root/'assessment-a');assert len(assessment['records'])==2
  second={**assessment_request,'reviewer_id':'reviewer-b','assessments':[{'item_id':'candidate:a','decision':'accept','confidence':.8},{'item_id':'candidate:b','decision':'uncertain','confidence':.5}]};m.record_assessment(second,root/'queue',root/'assessment-b')
  evaluation=m.evaluate({'assessments':[{'job_id':'a','artifact_id':'review-assessment'},{'job_id':'b','artifact_id':'review-assessment'}]},[root/'assessment-a',root/'assessment-b'],root/'evaluation');assert evaluation['record_count']==4;assert evaluation['roc_auc']==1.0;assert (root/'evaluation'/'evaluation-summary.png').is_file()
 print('ReviewAdapterTest passed');return 0
if __name__=='__main__':raise SystemExit(main())
