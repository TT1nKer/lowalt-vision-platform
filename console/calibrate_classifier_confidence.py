#!/usr/bin/env python3
"""Select a classifier confidence threshold with explicit abstention accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def calibrate(model_path,test_dir,output_path):
    model=YOLO(model_path);root=Path(test_dir).resolve();items=[]
    for truth in ('empty','occupied'):
        paths=sorted((root/truth).glob('*.jpg'))
        for start in range(0,len(paths),256):
            results=model.predict([str(p) for p in paths[start:start+256]],imgsz=128,device=0,verbose=False)
            items += [(truth,model.names[int(r.probs.top1)],float(r.probs.top1conf)) for r in results]
    reports={}
    for threshold in (.60,.70,.75,.80,.85,.90):
        decided=[x for x in items if x[2]>=threshold];unknown=len(items)-len(decided);metrics={}
        for cls in ('empty','occupied'):
            tp=sum(t==cls and p==cls for t,p,_ in decided);fp=sum(t!=cls and p==cls for t,p,_ in decided);fn=sum(t==cls and p!=cls for t,p,_ in decided)
            metrics[cls]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall_on_decided':tp/(tp+fn) if tp+fn else 0}
        reports[str(threshold)]={'decided':len(decided),'unknown':unknown,'unknown_rate':unknown/len(items),'accuracy_on_decided':sum(t==p for t,p,_ in decided)/len(decided),'metrics':metrics}
    eligible=[(float(k),v) for k,v in reports.items() if all(m['precision']>=.98 for m in v['metrics'].values())]
    recommended=min(eligible,key=lambda x:x[1]['unknown_rate'])[0] if eligible else .85
    result={'schema_version':1,'total':len(items),'thresholds':reports,'recommended_threshold':recommended,'policy':'minimize abstention while per-class decided precision >= 0.98'};Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(calibrate(a.model,a.test_dir,a.output),ensure_ascii=False,indent=2))
