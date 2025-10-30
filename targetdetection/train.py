#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : train.py
# Time       : 2025/10/20 20:34
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import random
import comet_ml
from pathlib import Path

import torch
from ultralytics import YOLO


# ksplit = 4
# split_dir = Path("/home/lab_NiT/huwl/TargetDetection/cleaned_data/2025-10-21_4-Fold_Cross-val")
# ds_yamls = []
# for s in range(1, ksplit+1):
#     split = f"split_{s}"
#     dataset_yaml = split_dir / split / f"{split}_dataset.yaml"
#     ds_yamls.append(dataset_yaml)

md = "./model.yaml"
ds = "./test/2025-10-29_2-Fold_Cross-val/split_1/split_1_dataset.yaml"

# results = {}
batch = 16
project = "DMV_Detection_YOLO"
epochs = 100
imgsz = 1024
device = [0, 1, 2, 3]
name = f"int8_b{batch}_e{epochs}_sz{imgsz}_model0"

comet_ml.login(project_name=project)

# for k, dataset_yaml in enumerate(ds_yamls):
#     # Load a model
#     model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#     results[k] = model.train(
#         data=dataset_yaml, epochs=epochs, batch=batch, project=project, name=f"int8_b16_e200_fold_{k + 1}", imgsz=imgsz, device=device
#     )  # include any additional train arguments

# task="detect" to handle NotImplementedError: 'YOLO' model does not support '_new' mode for 'None' task.
# model = YOLO(md, verbose=True, task="detect")
model = YOLO(md)

# Check model FLOPs. Failed forward pass causes 0 FLOPs.
# model.info()

# Inspect individual layers
# for i, layer in enumerate(model.model.model):
#     print(f"Layer {i}: {layer}")
#
# output = model.model(torch.randn(1, 3, 4096, 4096))
# print(f"Output shape: {output.shape}")  # Should match expected dimensions

result = model.train(data=ds, epochs=epochs, batch=batch, project=project, name=name, imgsz=imgsz, device=device)

