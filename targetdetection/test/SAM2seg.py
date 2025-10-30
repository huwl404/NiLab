#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : SAM2seg.py
# Time       : 2025/10/21 18:03
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
from pathlib import Path
from ultralytics import SAM

# Load a model
model = SAM("sam2.1_t.pt")

# Display model information (optional)
# model.info()
image = "/home/lab_NiT/huwl/TargetDetection/cleaned_data/map/mmm001_tile001.png"
txt = "/home/lab_NiT/huwl/TargetDetection/cleaned_data/label/mmm001_tile001.txt"

lines = Path(txt).read_text().splitlines()
# label == 0 means the samples are negative
labels = [1] * len(lines)
points = []
for line in lines:
    l = line.split(" ")
    l1 = float(l[1])
    l2 = float(l[2])
    points.append([l1 * 4096, l2 * 4096])

# Run inference
results = model(image, points=points, labels=labels)

# Display results
for result in results:
    result.show()
