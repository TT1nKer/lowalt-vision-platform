#!/usr/bin/env python3
"""Fine-grained Gemma audit using enlarged, numbered parking-space crop sheets."""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from training.build_topology_review import stamp
from console.core import load_config, safe_write_json
from console.golden_gemma_review import _call_gemma
from training.topology_inference import topology


PROMPT = """Audit parking-space state predictions using two contact sheets with identical numbered cells.

Sheet 1 is CLEAN CROPS. Sheet 2 is PREDICTIONS.
Prediction border/label: RED O=occupied, GREEN E=empty, GRAY ?=unknown.
Judge only the numbered cells. A crop is occupied when a vehicle substantially occupies that parking space. Ignore small fragments of adjacent vehicles at crop edges. Unknown is a safe abstention and is not an error.

Return ONLY JSON:
{{
  "verdict": "pass" | "needs_review" | "fail",
  "confidence": 0.0,
  "issues": ["wrong_class" | "other"],
  "evidence": "brief summary",
  "estimated_problem_count": 0,
  "wrong_ids": [0],
  "uncertain_ids": [0]
}}
"""


def _nearest(name, refs):
    s=stamp(name); candidates=refs.get(s[0],[]) if s else []
    return min((abs((s[1]-moment).total_seconds()),path) for moment,path in candidates)[1] if candidates else None


def _cell(crop, index, state=None, confidence=0):
    cell=np.full((170,150,3),235,dtype=np.uint8); resized=cv2.resize(crop,(138,138));cell[26:164,6:144]=resized
    cv2.putText(cell,f"ID {index}",(7,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,25,34),1,cv2.LINE_AA)
    if state:
        color={'occupied':(45,70,230),'empty':(45,190,80),'unknown':(150,150,150)}[state]
        cv2.rectangle(cell,(4,24),(145,165),color,3);cv2.putText(cell,f"{state[0].upper() if state!='unknown' else '?'} {confidence:.2f}",(70,18),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return cell


def _sheet(cells):
    blank=np.full((170,150,3),235,dtype=np.uint8);cells=cells+[blank]*(20-len(cells))
    return np.vstack([np.hstack(cells[i:i+5]) for i in range(0,20,5)])


def run(project_dir,config_path,model_path,source_dir,label_dir,shadow_dir,output_dir,images=8,workers=2):
    root=Path(project_dir).resolve();source=Path(source_dir).resolve();labels=Path(label_dir).resolve();shadow=Path(shadow_dir).resolve();output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(config_path);model=YOLO(model_path);refs={}
    for path in labels.glob('*.txt'):
        s=stamp(path.name)
        if s:refs.setdefault(s[0],[]).append((s[1],path))
    risk_path=shadow/'gemma_business_review.jsonl';risk=[]
    for line in risk_path.read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        if item.get('verdict')=='fail':risk.append(item)
    risk.sort(key=lambda x:(int(x.get('critical_fp',0)),int(x.get('estimated_problem_count',0))),reverse=True);risk=risk[:images]
    tasks=[];manifest=[]
    for item in risk:
        ref=_nearest(item['image'],refs)
        if not ref:continue
        image=cv2.imread(str(source/item['image']));spaces=topology(ref,image.shape[1],image.shape[0]);crops=[]
        for box in spaces:
            x1,y1,x2,y2=box;dx=(x2-x1)*.08;dy=(y2-y1)*.08;x1,x2=int(x1+dx),int(x2-dx);y1,y2=int(y1+dy),int(y2-dy);crops.append(image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)])
        results=model.predict(crops,imgsz=128,device=0,verbose=False);pred=[]
        for result in results:
            conf=float(result.probs.top1conf);pred.append((model.names[int(result.probs.top1)] if conf>=.85 else 'unknown',conf))
        image_dir=output/Path(item['image']).stem;image_dir.mkdir(exist_ok=True)
        for page,start in enumerate(range(0,len(crops),20),1):
            clean=_sheet([_cell(crops[i],i) for i in range(start,min(start+20,len(crops)))])
            marked=_sheet([_cell(crops[i],i,*pred[i]) for i in range(start,min(start+20,len(crops)))])
            clean_path=image_dir/f'page_{page}_clean.jpg';marked_path=image_dir/f'page_{page}_pred.jpg';cv2.imwrite(str(clean_path),clean,[cv2.IMWRITE_JPEG_QUALITY,94]);cv2.imwrite(str(marked_path),marked,[cv2.IMWRITE_JPEG_QUALITY,94])
            tasks.append((item['image'],page,clean_path,marked_path,start,min(start+20,len(crops))))
        manifest.append({'image':item['image'],'reference':ref.name,'spaces':len(spaces),'source_gemma':item})
    journal=output/'gemma_sheet_review.jsonl';completed=set()
    if journal.is_file():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:x=json.loads(line);completed.add((x['image'],x['page']))
            except Exception:pass
    lock=threading.Lock()
    def audit(task):
        image,page,clean,marked,start,end=task
        try:
            parsed,raw=_call_gemma(cfg,PROMPT+f"\nThis page contains IDs {start} through {end-1}.",clean,marked)
            valid=set(range(start,end));parsed['wrong_ids']=[int(x) for x in parsed.get('wrong_ids',[]) if str(x).isdigit() and int(x) in valid];parsed['uncertain_ids']=[int(x) for x in parsed.get('uncertain_ids',[]) if str(x).isdigit() and int(x) in valid]
            return {'status':'completed','image':image,'page':page,'range':[start,end],'reviewed_at':datetime.now().isoformat(),**parsed,'raw':raw}
        except Exception as exc:return {'status':'failed','image':image,'page':page,'range':[start,end],'reviewed_at':datetime.now().isoformat(),'error':str(exc)[:1000]}
    pending=[t for t in tasks if (t[0],t[1]) not in completed]
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        futures=[pool.submit(audit,t) for t in pending]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            with lock,journal.open('a',encoding='utf-8') as handle:handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(f"@@PROGRESS {i} {len(pending)} {result['image']} page={result['page']}",flush=True)
    latest=[]
    for line in journal.read_text(encoding='utf-8').splitlines():latest.append(json.loads(line))
    summary={'schema_version':1,'generated_at':datetime.now().isoformat(),'images':len(manifest),'pages':len(tasks),'completed_pages':sum(x.get('status')=='completed' for x in latest),'wrong_predictions':sum(len(x.get('wrong_ids',[])) for x in latest),'uncertain_predictions':sum(len(x.get('uncertain_ids',[])) for x in latest),'manifest':manifest,'human_approval_created':False}
    safe_write_json(str(output/'summary.json'),summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-dir',default='.');p.add_argument('--config',default='config.yaml');p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--labels',required=True);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);p.add_argument('--images',type=int,default=8);p.add_argument('--workers',type=int,default=2);a=p.parse_args();print(json.dumps(run(a.project_dir,a.config,a.model,a.source,a.labels,a.shadow,a.output,a.images,a.workers),ensure_ascii=False,indent=2))
