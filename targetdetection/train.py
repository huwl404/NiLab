#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : train.py
# Time       : 2025/10/20 20:34
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import comet_ml
from pathlib import Path
from ultralytics import YOLO


ksplit = 3
split_dir = Path("/home/lab_NiT/huwl/TargetDetection/cleaned_data/2025-10-21_3-Fold_Cross-val")
ds_yamls = []
for s in range(1, ksplit+1):
    split = f"split_{s}"
    dataset_yaml = split_dir / split / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

# Train the model
results = {}
batch = 16
project = "3fold"
epochs = 200
imgsz = 2048
device = [0, 1, 2, 3]

comet_ml.login(project_name=project)
for k, dataset_yaml in enumerate(ds_yamls):
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results[k] = model.train(
        data=dataset_yaml, epochs=epochs, batch=batch, project=project, name=f"int8_b16_e200_fold_{k + 1}", imgsz=imgsz, device=device
    )  # include any additional train arguments
