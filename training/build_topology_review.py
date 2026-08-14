#!/usr/bin/env python3
"""Overlay monitored-space topology on calibrated shadow images for Gemma review."""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, time
from pathlib import Path

import cv2
import numpy as np


STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})")


def stamp(name):
    m=STAMP.match(name)
    return (m.group(1),datetime.combine(date.min,time(*map(int,m.groups()[1:])))) if m else None


def build(shadow_dir: str,label_dir: str) -> dict:
    root=Path(shadow_dir).resolve(); labels=Path(label_dir).resolve()
    report=json.loads((root/'shadow_report.json').read_text(encoding='utf-8')); refs={}
    for path in labels.glob('*.txt'):
        s=stamp(path.name)
        if s: refs.setdefault(s[0],[]).append((s[1],path))
    output=root/'topology_review';output.mkdir(exist_ok=True);records=[]
    for index,row in enumerate(report['records'][:30],1):
        s=stamp(row['image']); candidates=refs.get(s[0],[]) if s else []
        if not candidates: continue
        _,ref=min((abs((s[1]-moment).total_seconds()),path) for moment,path in candidates)
        image=cv2.imread(str(root/row['comparison'])); h,w=image.shape[:2]; panel_w=(w-4)//2
        count=0
        for raw in ref.read_text(encoding='utf-8').splitlines():
            z=raw.split()
            if len(z)!=5 or int(z[0]) not in {1,2}: continue
            cx,cy,bw,bh=map(float,z[1:]);x1=int(panel_w+(cx-bw/2)*panel_w+4);y1=int((cy-bh/2)*h);x2=int(panel_w+(cx+bw/2)*panel_w+4);y2=int((cy+bh/2)*h)
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,215,255),1,cv2.LINE_AA);count+=1
        cv2.putText(image,f"YELLOW=monitored topology ({count} spaces)",(panel_w+12,h-15),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,215,255),2,cv2.LINE_AA)
        name=f"topology_{index:03d}.jpg";cv2.imwrite(str(output/name),image,[cv2.IMWRITE_JPEG_QUALITY,92])
        records.append({"image":row['image'],"comparison":name,"source_comparison":row['comparison'],"reference":ref.name,"monitored_spaces":count,"baseline":row['baseline'],"candidate":row['candidate']})
    derived={**report,"status":"gemma_topology_review","records":records,"review_contract":"Only yellow monitored topology spaces are in scope."}
    (output/'shadow_report.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    return {"output":str(output),"records":len(records)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shadow-dir',default='quality/business_shadow_calibrated_v1');p.add_argument('--labels',default=r'H:\pklotdataset\test\labels');a=p.parse_args();print(json.dumps(build(a.shadow_dir,a.labels),ensure_ascii=False,indent=2))
