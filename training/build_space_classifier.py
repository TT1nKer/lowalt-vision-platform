#!/usr/bin/env python3
"""Build a date-isolated occupied/empty crop dataset from official PKLot labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2


def rank(name: str) -> str:
    return hashlib.sha256(("space-classifier-v1:"+name).encode()).hexdigest()


def build(source_dir: str, output_dir: str, train_frames=300, val_frames=80, test_frames=100, padding=.12):
    source=Path(source_dir).resolve();output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError(f"classifier dataset already exists: {output}")
    limits={'train':train_frames,'val':val_frames,'test':test_frames};counts=Counter();frames={}
    staging=output.with_name(output.name+'.staging')
    try:
        for split in limits:
            for cls in ('occupied','empty'):(staging/split/cls).mkdir(parents=True,exist_ok=True)
            images=sorted((source/'images'/split).glob('*.jpg'),key=lambda p:rank(p.name))[:limits[split]];frames[split]=[p.name for p in images]
            for frame_index,image_path in enumerate(images):
                label=source/'labels'/split/(image_path.stem+'.txt')
                image=cv2.imread(str(image_path));height,width=image.shape[:2]
                for object_index,raw in enumerate(label.read_text(encoding='utf-8').splitlines()):
                    z=raw.split()
                    if len(z)!=9 or int(z[0]) not in {0,1}:continue
                    pts=list(map(float,z[1:]));xs=pts[0::2];ys=pts[1::2];x1,x2=min(xs),max(xs);y1,y2=min(ys),max(ys);bw=x2-x1;bh=y2-y1
                    x1=max(0,int((x1-padding*bw)*width));x2=min(width,int((x2+padding*bw)*width));y1=max(0,int((y1-padding*bh)*height));y2=min(height,int((y2+padding*bh)*height))
                    if x2-x1<4 or y2-y1<4:continue
                    cls='occupied' if int(z[0])==0 else 'empty';crop=image[y1:y2,x1:x2]
                    name=f"{image_path.stem}__{object_index:03d}.jpg";cv2.imwrite(str(staging/split/cls/name),crop,[cv2.IMWRITE_JPEG_QUALITY,90]);counts[f'{split}_{cls}']+=1
        manifest={'schema_version':1,'created_at':datetime.now().isoformat(),'source':str(source),'padding':padding,'frame_limits':limits,'frames':frames,'counts':dict(counts),'split_contract':'inherits capture-day isolated official recovery v2 splits'}
        (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');staging.rename(output);return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='quality/pklot_official_recovery_v2');p.add_argument('--output',default='quality/space_classifier_v1');p.add_argument('--train-frames',type=int,default=300);p.add_argument('--val-frames',type=int,default=80);p.add_argument('--test-frames',type=int,default=100);p.add_argument('--padding',type=float,default=.12);a=p.parse_args();print(json.dumps(build(a.source,a.output,a.train_frames,a.val_frames,a.test_frames,a.padding),ensure_ascii=False,indent=2))
