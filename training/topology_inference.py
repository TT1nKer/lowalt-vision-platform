#!/usr/bin/env python3
"""Project detector output onto fixed monitored parking-space topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from training.build_topology_review import stamp


def iou(a,b):
    tl=np.maximum(a[:2],b[:2]);br=np.minimum(a[2:],b[2:]);inter=np.maximum(br-tl,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return inter/max(aa+bb-inter,1e-9)


def topology(path,width,height):
    boxes=[]
    for raw in path.read_text(encoding='utf-8').splitlines():
        z=raw.split()
        if len(z)!=5 or int(z[0]) not in {1,2}:continue
        cx,cy,w,h=map(float,z[1:]);boxes.append(np.array([(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height],dtype=np.float32))
    return boxes


def run(model_path,source_dir,label_dir,shadow_dir,output_dir,conf=.25,min_iou=.30,min_class_conf=.65):
    source=Path(source_dir).resolve();labels=Path(label_dir).resolve();shadow=Path(shadow_dir).resolve();output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    base=json.loads((shadow/'shadow_report.json').read_text(encoding='utf-8'));refs={}
    for p in labels.glob('*.txt'):
        s=stamp(p.name)
        if s:refs.setdefault(s[0],[]).append((s[1],p))
    model=YOLO(model_path);records=[];aggregate={'occupied':0,'empty':0,'unknown':0}
    for index,row in enumerate(base['records'][:30],1):
        s=stamp(row['image']); candidates=refs.get(s[0],[]) if s else []
        if not candidates:continue
        _,ref=min((abs((s[1]-moment).total_seconds()),path) for moment,path in candidates)
        image_path=source/row['image'];image=cv2.imread(str(image_path));result=model.predict(str(image_path),imgsz=640,conf=conf,device=0,verbose=False)[0]
        spaces=topology(ref,image.shape[1],image.shape[0]);pred=[]
        if result.obb is not None:
            for box,cls,score in zip(result.obb.xyxy.cpu().numpy(),result.obb.cls.cpu().numpy(),result.obb.conf.cpu().numpy()):pred.append((box,int(cls),float(score)))
        states=[]
        for space in spaces:
            candidates2=sorted(((iou(space,p[0]),p) for p in pred),key=lambda x:(x[0],x[1][2]),reverse=True)
            if candidates2 and candidates2[0][0]>=min_iou and candidates2[0][1][2]>=min_class_conf:state='occupied' if candidates2[0][1][1]==0 else 'empty';score=candidates2[0][1][2]
            else:state='unknown';score=candidates2[0][1][2] if candidates2 else 0
            states.append(state);aggregate[state]+=1
            x1,y1,x2,y2=space.astype(int);color={'occupied':(45,70,230),'empty':(45,190,80),'unknown':(150,150,150)}[state];cv2.rectangle(image,(x1,y1),(x2,y2),color,2,cv2.LINE_AA);cv2.putText(image,{'occupied':'O','empty':'E','unknown':'?'}[state],(x1,max(45,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
        cv2.rectangle(image,(0,0),(image.shape[1],40),(20,25,34),-1);cv2.putText(image,f"STATES {min_class_conf:.2f}",(12,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA);cv2.putText(image,"RED=occupied  GREEN=empty  GRAY=unknown",(180,26),cv2.FONT_HERSHEY_SIMPLEX,.33,(220,220,220),1,cv2.LINE_AA)
        name=f"topology_state_{index:03d}.jpg";cv2.imwrite(str(output/name),image,[cv2.IMWRITE_JPEG_QUALITY,92]);counts={k:states.count(k) for k in aggregate};records.append({'image':row['image'],'comparison':name,'reference':ref.name,'candidate':{'detections':len(spaces),'classes':counts},'baseline':row['baseline']});print(f"@@PROGRESS {index} 30 {row['image']}",flush=True)
    report={**base,'status':'topology_state_candidate','records':records,'aggregate':{'candidate':aggregate},'inference':{'detector_confidence':conf,'class_confidence':min_class_conf,'min_topology_iou':min_iou},'review_contract':'Exactly one occupied/empty/unknown state per fixed monitored space.'};(output/'shadow_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--labels',required=True);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.model,a.source,a.labels,a.shadow,a.output);print(json.dumps({'records':len(r['records']),'aggregate':r['aggregate'],'inference':r['inference']},ensure_ascii=False,indent=2))
