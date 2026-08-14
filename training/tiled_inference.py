#!/usr/bin/env python3
"""Run overlap-tiled OBB inference and merge detections in full-image coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COLORS = {0: (45, 70, 230), 1: (45, 190, 80)}


def _iou(a, b):
    tl = np.maximum(a[:2], b[:2]); br = np.minimum(a[2:], b[2:])
    inter = np.maximum(br-tl, 0).prod(); aa=np.maximum(a[2:]-a[:2],0).prod(); bb=np.maximum(b[2:]-b[:2],0).prod()
    return inter / max(aa+bb-inter, 1e-9)


def predict(model, image: np.ndarray, *, rows=2, cols=2, overlap=.20, imgsz=640, conf=.55):
    height, width = image.shape[:2]; detections=[]
    tile_w = int(np.ceil(width / (cols - (cols-1)*overlap)))
    tile_h = int(np.ceil(height / (rows - (rows-1)*overlap)))
    xs = np.linspace(0, max(0, width-tile_w), cols).astype(int); ys=np.linspace(0,max(0,height-tile_h),rows).astype(int)
    for y in ys:
        for x in xs:
            crop=image[y:min(y+tile_h,height),x:min(x+tile_w,width)]
            result=model.predict(crop,imgsz=imgsz,conf=conf,device=0,verbose=False)[0]
            if result.obb is None: continue
            for polygon, cls, score in zip(result.obb.xyxyxyxy.cpu().numpy(),result.obb.cls.cpu().numpy(),result.obb.conf.cpu().numpy()):
                polygon=polygon+np.array([x,y],dtype=np.float32); bounds=np.r_[polygon.min(0),polygon.max(0)]
                detections.append({"polygon":polygon,"bounds":bounds,"class_id":int(cls),"confidence":float(score)})
    detections.sort(key=lambda d:d["confidence"],reverse=True); kept=[]
    for item in detections:
        if any(item["class_id"]==other["class_id"] and _iou(item["bounds"],other["bounds"])>.45 for other in kept): continue
        kept.append(item)
    return kept


def render(model_path: str,image_path: str,output_dir: str):
    model=YOLO(model_path); image=cv2.imread(str(Path(image_path).resolve())); output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    variants=[(2,2,.20,.65),(2,2,.20,.50),(3,2,.20,.60),(3,2,.20,.45)]; records=[]
    for rows,cols,overlap,conf in variants:
        det=predict(model,image,rows=rows,cols=cols,overlap=overlap,imgsz=640,conf=conf); canvas=image.copy()
        cv2.rectangle(canvas,(0,0),(canvas.shape[1],40),(20,25,34),-1);cv2.putText(canvas,f"TILED {rows}x{cols} conf={conf:.2f}",(12,27),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA)
        for d in det:
            color=COLORS.get(d["class_id"],(255,255,255));cv2.polylines(canvas,[d["polygon"].astype(np.int32).reshape(-1,1,2)],True,color,2,cv2.LINE_AA)
        name=f"tile_{rows}x{cols}_{int(conf*100)}.jpg";cv2.imwrite(str(output/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
        records.append({"rows":rows,"cols":cols,"overlap":overlap,"confidence":conf,"detections":len(det),"occupied":sum(d['class_id']==0 for d in det),"empty":sum(d['class_id']==1 for d in det),"file":name})
    combined=np.hstack([cv2.imread(str(output/r['file'])) for r in records]);cv2.imwrite(str(output/'tiled_all.jpg'),combined,[cv2.IMWRITE_JPEG_QUALITY,92])
    report={"image":Path(image_path).name,"variants":records};(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(render(a.model,a.image,a.output),ensure_ascii=False,indent=2))
