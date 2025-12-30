#!/usr/bin/env python
# -*- coding:utf-8 -*-

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./data/11n_1K_classifier_best.pt")

# Define a glob search for all JPG files in a directory
source = "./test/*.png"

# Run inference on the source
results = model(source,
                stream=True,
                imgsz=1024,
                device="cpu",
                project="whereisproject",
                name="whereisname")

for r in results:
    if r is not None:
        print(r)
        p = r.probs
        cls = r.names[p.top1]
        conf = round(p.top1conf.item(), 2)
        print(f"{cls}: {conf}")