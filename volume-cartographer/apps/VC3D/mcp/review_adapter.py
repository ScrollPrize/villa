#!/usr/bin/env python3
"""Immutable local review queues, assessments, and labeled evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

MAX_QUEUE_ITEMS = 100
MAX_ASSESSMENTS = 100
MAX_ASSESSMENT_ARTIFACTS = 16
DECISIONS = {"accept", "reject", "uncertain", "defer"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError(f"expected JSON object in {path}")
    return value


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: dict) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bar_chart(path: Path, labels: list[str], values: list[float], width: int = 820) -> None:
    from PIL import Image, ImageDraw
    row=38;height=max(90,36+row*len(labels));image=Image.new("RGB",(width,height),(23,21,18));draw=ImageDraw.Draw(image)
    for index,(label,value) in enumerate(zip(labels,values)):
        y=18+index*row;draw.text((12,y+6),label[:28],fill=(240,231,210));draw.rectangle((230,y+3,width-18,y+26),outline=(57,51,41));end=231+int((width-250)*max(0,min(1,value)));draw.rectangle((231,y+4,end,y+25),fill=(242,169,59));draw.text((width-62,y+6),f"{value:.3f}",fill=(240,231,210))
    image.save(path)


def create_queue(request: dict, ranking_path: Path, comparison_path: Path | None, output: Path) -> dict:
    ranking=read_json(ranking_path/'manifest.json');items=[]
    for candidate in ranking.get('ranked_candidates',[]):
        items.append({'item_id':f"candidate:{candidate['id']}",'kind':'candidate','priority':float(candidate['combined_score']),'candidate_id':candidate['id'],'rank':candidate['rank'],'components':candidate['components'],'source_artifact':request['ranking']})
    if comparison_path is not None:
        import zarr
        comparison=read_json(comparison_path/'manifest.json');group=zarr.open_group(str(comparison_path/'structural-comparison.zarr'),mode='r')
        priority=np.asarray(group['review_priority'],np.float32);divergence=np.asarray(group['divergence'],np.float32);xs=np.asarray(group['centers_u'],np.int64);ys=np.asarray(group['centers_v'],np.int64)
        finite=np.isfinite(priority);percentile=float(request.get('divergence_percentile',90.0))
        if not 50<=percentile<=99.9: raise ValueError('divergence_percentile must be from 50 to 99.9')
        threshold=float(np.percentile(priority[finite],percentile)) if finite.any() else math.inf
        labels,count=ndimage.label(finite&(priority>=threshold),structure=np.ones((3,3),np.uint8));scale=max(float(np.percentile(priority[finite],95)),1e-9) if finite.any() else 1
        regions=[]
        for label in range(1,count+1):
            rows,cols=np.where(labels==label)
            if not rows.size:continue
            local=priority[rows,cols];best=int(np.argmax(local));regions.append({'raw_priority':float(local[best]),'mean_raw_priority':float(local.mean()),'grid_bbox':{'u_index_min':int(cols.min()),'v_index_min':int(rows.min()),'u_index_max':int(cols.max()),'v_index_max':int(rows.max())},'center_uv':{'u':int(xs[cols[best]]),'v':int(ys[rows[best]])},'mean_divergence':float(divergence[rows,cols].mean()),'window_count':int(rows.size)})
        regions.sort(key=lambda x:-x['raw_priority'])
        for index,region in enumerate(regions,1):
            items.append({'item_id':f'divergence:{index:03d}','kind':'divergence_region','priority':float(np.clip(region['raw_priority']/scale,0,1)),'region':region,'source_artifact':request['comparison']})
    items.sort(key=lambda item:(-item['priority'],item['item_id']));maximum=int(request.get('max_items',50))
    if not 1<=maximum<=MAX_QUEUE_ITEMS:raise ValueError('max_items must be from 1 to 100')
    items=items[:maximum]
    core={'kind':'vc_review_queue_v1','score_semantics':'prioritized_human_review_not_truth','source_ranking':request['ranking'],'source_comparison':request.get('comparison'),'created_at':now(),'items':items}
    core['queue_digest']=digest({key:value for key,value in core.items() if key!='created_at'});output.mkdir(parents=True,exist_ok=True)
    (output/'review-queue.json').write_text(json.dumps(core,indent=2)+'\n');(output/'manifest.json').write_text(json.dumps(core,indent=2)+'\n')
    with (output/'review-queue.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['item_id','kind','priority']);writer.writerows((item['item_id'],item['kind'],item['priority']) for item in items)
    bar_chart(output/'review-queue.png',[item['item_id'] for item in items],[item['priority'] for item in items])
    core['queue_preview']='review-queue.png';(output/'manifest.json').write_text(json.dumps(core,indent=2)+'\n');return core


def record_assessment(request: dict, queue_path: Path, output: Path) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,64}',str(request.get('reviewer_id',''))):raise ValueError('reviewer_id contains unsupported characters')
    queue=read_json(queue_path/'review-queue.json');by_id={item['item_id']:item for item in queue['items']};records=[];seen=set()
    for assessment in request['assessments']:
        item_id=str(assessment['item_id']);decision=str(assessment['decision'])
        if item_id not in by_id:raise ValueError(f'queue item not found: {item_id}')
        if item_id in seen:raise ValueError(f'duplicate assessment item: {item_id}')
        if decision not in DECISIONS:raise ValueError(f'unsupported decision: {decision}')
        seen.add(item_id);confidence=float(assessment.get('confidence',0.5))
        if not 0<=confidence<=1:raise ValueError('assessment confidence must be from 0 to 1')
        reasons=[str(reason) for reason in assessment.get('reason_codes',[])]
        if len(reasons)>8 or any(not re.fullmatch(r'[A-Za-z0-9._-]{1,64}',reason) for reason in reasons):raise ValueError('reason codes must contain at most 8 safe bounded values')
        notes=str(assessment.get('notes',''))
        if len(notes)>1000:raise ValueError('assessment notes exceed 1000 characters')
        item=by_id[item_id];records.append({'item_id':item_id,'kind':item['kind'],'priority':item['priority'],'decision':decision,'confidence':confidence,'reason_codes':reasons,'notes':notes})
    document={'kind':'vc_review_assessment_v1','label_semantics':'reviewer_assessment_not_universal_ground_truth','queue_artifact':request['queue'],'queue_digest':queue['queue_digest'],'reviewer_id':request['reviewer_id'],'created_at':now(),'records':records}
    document['assessment_digest']=digest({key:value for key,value in document.items() if key!='created_at'});output.mkdir(parents=True,exist_ok=True)
    (output/'review-assessment.json').write_text(json.dumps(document,indent=2)+'\n');(output/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    counts={decision:sum(record['decision']==decision for record in records) for decision in sorted(DECISIONS)};bar_chart(output/'assessment-summary.png',list(counts),[counts[key]/max(1,len(records)) for key in counts]);document['assessment_preview']='assessment-summary.png';(output/'manifest.json').write_text(json.dumps(document,indent=2)+'\n');return document


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive=scores[labels==1];negative=scores[labels==0]
    if not len(positive) or not len(negative):return None
    return float((sum((p>negative).sum()+.5*(p==negative).sum() for p in positive))/(len(positive)*len(negative)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if labels.sum()==0:return None
    order=np.argsort(-scores,kind='stable');ordered=labels[order];precision=np.cumsum(ordered)/(np.arange(len(ordered))+1);return float((precision*ordered).sum()/ordered.sum())


def evaluate(request: dict, assessment_paths: list[Path], output: Path) -> dict:
    if not 1<=len(assessment_paths)<=MAX_ASSESSMENT_ARTIFACTS:raise ValueError('evaluation requires from 1 to 16 assessment artifacts')
    documents=[read_json(path/'review-assessment.json') for path in assessment_paths];queue_digests={doc['queue_digest'] for doc in documents}
    if len(queue_digests)!=1:raise ValueError('all assessments must refer to the same immutable review queue')
    rows=[]
    for doc in documents:
        for record in doc['records']:rows.append({**record,'reviewer_id':doc['reviewer_id']})
    binary=[row for row in rows if row['decision'] in ('accept','reject')];scores=np.asarray([row['priority'] for row in binary],float);labels=np.asarray([row['decision']=='accept' for row in binary],np.int8)
    auc=binary_auc(scores,labels);ap=average_precision(scores,labels)
    bins=[]
    for low in np.linspace(0,0.8,5):
        selected=[row for row in binary if low<=row['priority']<low+.2 or (low==.8 and row['priority']==1)]
        bins.append({'priority_min':float(low),'priority_max':float(low+.2),'count':len(selected),'accept_fraction':float(sum(row['decision']=='accept' for row in selected)/len(selected)) if selected else None})
    grouped={}
    for row in rows:grouped.setdefault(row['item_id'],[]).append(row)
    pairs=agree=0;kappas=[]
    reviewer_ids=sorted({doc['reviewer_id'] for doc in documents})
    for i,a in enumerate(reviewer_ids):
        for b in reviewer_ids[i+1:]:
            paired=[]
            for values in grouped.values():
                da=next((v['decision'] for v in values if v['reviewer_id']==a and v['decision'] in ('accept','reject')),None);db=next((v['decision'] for v in values if v['reviewer_id']==b and v['decision'] in ('accept','reject')),None)
                if da and db:paired.append((da,db))
            if paired:
                observed=sum(x==y for x,y in paired)/len(paired);pa=sum(x=='accept' for x,_ in paired)/len(paired);pb=sum(y=='accept' for _,y in paired)/len(paired);expected=pa*pb+(1-pa)*(1-pb);kappa=(observed-expected)/(1-expected) if expected<1 else 1.0;kappas.append({'reviewer_a':a,'reviewer_b':b,'overlap':len(paired),'agreement':observed,'cohen_kappa':kappa});pairs+=len(paired);agree+=sum(x==y for x,y in paired)
    decisions={decision:sum(row['decision']==decision for row in rows) for decision in sorted(DECISIONS)}
    manifest={'kind':'vc_review_label_evaluation_v1','metric_semantics':'evaluation_against_supplied_reviewer_labels_not_objective_truth','assessment_artifacts':request['assessments'],'queue_digest':next(iter(queue_digests)),'record_count':len(rows),'binary_record_count':len(binary),'decision_counts':decisions,'roc_auc':auc,'average_precision':ap,'calibration_bins':bins,'pairwise_reviewer_agreement':kappas,'overall_pair_agreement':float(agree/pairs) if pairs else None,'limitations':['accept/reject decisions are reviewer labels, not universal ground truth','repeated labels across reviewers are retained for reviewer-agreement analysis and pooled ranking metrics']}
    output.mkdir(parents=True,exist_ok=True);(output/'evaluation.json').write_text(json.dumps(manifest,indent=2)+'\n');(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    labels_chart=['ROC AUC','Average precision','Pair agreement'];values=[auc or 0,ap or 0,manifest['overall_pair_agreement'] or 0];bar_chart(output/'evaluation-summary.png',labels_chart,values);manifest['evaluation_preview']='evaluation-summary.png';(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (output/'evaluation-records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['reviewer_id','item_id','kind','priority','decision','confidence']);writer.writeheader();writer.writerows({key:row[key] for key in writer.fieldnames} for row in rows)
    return manifest


def main()->int:
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('create');create.add_argument('--request',required=True,type=Path);create.add_argument('--ranking',required=True,type=Path);create.add_argument('--comparison',type=Path);create.add_argument('--output',required=True,type=Path)
    assess=commands.add_parser('assess');assess.add_argument('--request',required=True,type=Path);assess.add_argument('--queue',required=True,type=Path);assess.add_argument('--output',required=True,type=Path)
    evaluation=commands.add_parser('evaluate');evaluation.add_argument('--request',required=True,type=Path);evaluation.add_argument('--assessments',required=True,type=Path);evaluation.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();request=read_json(args.request);args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='create':result=create_queue(request,args.ranking,args.comparison,args.output)
    elif args.command=='assess':result=record_assessment(request,args.queue,args.output)
    else:result=evaluate(request,[Path(path) for path in read_json(args.assessments)['paths']],args.output)
    print(json.dumps(result),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
