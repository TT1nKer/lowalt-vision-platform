#!/usr/bin/env python3
"""Evaluate the space classifier on the fixed date-isolated test crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(model_path: str, test_dir: str, output_path: str):
    model=YOLO(model_path);root=Path(test_dir).resolve();names=model.names;confusion={truth:{pred:0 for pred in ('empty','occupied')} for truth in ('empty','occupied')}
    errors=[];total=0
    for truth in ('empty','occupied'):
        paths=sorted((root/truth).glob('*.jpg'))
        for start in range(0,len(paths),256):
            results=model.predict([str(p) for p in paths[start:start+256]],imgsz=128,device=0,verbose=False)
            for path,result in zip(paths[start:start+256],results):
                pred=names[int(result.probs.top1)];confusion[truth][pred]+=1;total+=1
                if pred!=truth and len(errors)<100:errors.append({'image':path.name,'truth':truth,'predicted':pred,'confidence':float(result.probs.top1conf)})
    metrics={}
    for cls in ('empty','occupied'):
        tp=confusion[cls][cls];fp=sum(confusion[t][cls] for t in confusion if t!=cls);fn=sum(confusion[cls][p] for p in confusion[cls] if p!=cls)
        metrics[cls]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0}
    report={'schema_version':1,'model':str(Path(model_path).resolve()),'test_dir':str(root),'total':total,'accuracy':sum(confusion[c][c] for c in confusion)/total,'confusion':confusion,'metrics':metrics,'sample_errors':errors,'status':'passed' if all(m['precision']>=.90 and m['recall']>=.85 for m in metrics.values()) else 'failed'}
    Path(output_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(evaluate(a.model,a.test_dir,a.output),ensure_ascii=False,indent=2))
