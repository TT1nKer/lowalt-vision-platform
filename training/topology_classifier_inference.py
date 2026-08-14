#!/usr/bin/env python3
"""Classify each fixed monitored parking space with the dedicated crop classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from training.build_topology_review import stamp
from training.topology_inference import topology


def run(model_path,source_dir,label_dir,shadow_dir,output_dir,min_conf=.85,shrink=.20):
    source=Path(source_dir).resolve();labels=Path(label_dir).resolve();shadow=Path(shadow_dir).resolve();output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    base=json.loads((shadow/'shadow_report.json').read_text(encoding='utf-8'));refs={}
    for p in labels.glob('*.txt'):
        s=stamp(p.name)
        if s:refs.setdefault(s[0],[]).append((s[1],p))
    model=YOLO(model_path);records=[];aggregate={'occupied':0,'empty':0,'unknown':0}
    for index,row in enumerate(base['records'][:30],1):
        s=stamp(row['image']); candidates=refs.get(s[0],[]) if s else []
        if not candidates:continue
        _,ref=min((abs((s[1]-moment).total_seconds()),path) for moment,path in candidates);image=cv2.imread(str(source/row['image']));spaces=topology(ref,image.shape[1],image.shape[0]);crops=[]
        for box in spaces:
            x1,y1,x2,y2=box;dx=(x2-x1)*shrink;dy=(y2-y1)*shrink;x1,x2=int(x1+dx),int(x2-dx);y1,y2=int(y1+dy),int(y2-dy);crops.append(image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)])
        results=model.predict(crops,imgsz=128,device=0,verbose=False);states=[]
        for space,result in zip(spaces,results):
            confidence=float(result.probs.top1conf);state=model.names[int(result.probs.top1)] if confidence>=min_conf else 'unknown';states.append(state);aggregate[state]+=1;x1,y1,x2,y2=space.astype(int);color={'occupied':(45,70,230),'empty':(45,190,80),'unknown':(150,150,150)}[state];cv2.rectangle(image,(x1,y1),(x2,y2),color,2,cv2.LINE_AA);cv2.putText(image,{'occupied':'O','empty':'E','unknown':'?'}[state],(x1,max(43,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
        cv2.rectangle(image,(0,0),(image.shape[1],40),(20,25,34),-1);cv2.putText(image,f"SPACE CLASSIFIER {min_conf:.2f}",(12,26),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2,cv2.LINE_AA);cv2.putText(image,"RED=occupied GREEN=empty GRAY=unknown",(260,26),cv2.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1,cv2.LINE_AA)
        name=f"classified_{index:03d}.jpg";cv2.imwrite(str(output/name),image,[cv2.IMWRITE_JPEG_QUALITY,92]);counts={k:states.count(k) for k in aggregate};records.append({'image':row['image'],'comparison':name,'reference':ref.name,'candidate':{'detections':len(spaces),'classes':counts},'baseline':row['baseline']});print(f"@@PROGRESS {index} 30 {row['image']}",flush=True)
    report={**base,'status':'dedicated_space_classifier_candidate','records':records,'aggregate':{'candidate':aggregate},'inference':{'classifier_confidence':min_conf,'crop_shrink':shrink},'model':str(Path(model_path).resolve()),'review_contract':'Fixed monitored topology plus dedicated occupied/empty classifier; unknown is safe abstention.'};(output/'shadow_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--labels',required=True);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);p.add_argument('--min-conf',type=float,default=.85);a=p.parse_args();r=run(a.model,a.source,a.labels,a.shadow,a.output,a.min_conf);print(json.dumps({'records':len(r['records']),'aggregate':r['aggregate']},ensure_ascii=False,indent=2))
