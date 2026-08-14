#!/usr/bin/env python3
"""Train a low-cost parking-space state classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def train(data: str, model: str, project: str, epochs=5, transfer: str = ""):
    trainer=YOLO(model)
    if transfer:
        trainer.load(transfer)
    result=trainer.train(data=str(Path(data).resolve()),epochs=epochs,imgsz=128,batch=128,device=0,workers=0,project=str(Path(project).resolve()),name='space_cls_v1',exist_ok=False,patience=3,cache=False,verbose=True)
    return {'save_dir':str(result.save_dir),'epochs':epochs,'model':model,'transfer':transfer}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='quality/space_classifier_v1');p.add_argument('--model',default='yolo11n-cls.yaml');p.add_argument('--transfer',default='');p.add_argument('--project',default='quality/space_classifier_runs');p.add_argument('--epochs',type=int,default=5);a=p.parse_args();print(json.dumps(train(a.data,a.model,a.project,a.epochs,a.transfer),ensure_ascii=False,indent=2))
