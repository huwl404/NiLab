#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : tune.py
# Time       : 2025/10/31 15:37
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
import comet_ml
from ultralytics import YOLO

# Define search space
search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 100.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1, 10),
    "cls": (0.2, 4.0),
    "hsv_h": (0, 1),
    "hsv_s": (0, 1),
    "hsv_v": (0, 1),
    "degrees": (0, 180),
    "translate": (0, 0.9),
    "scale": (0, 0.9),
    "shear": (0, 10),
    "flipud": (0, 1),
    "fliplr": (0, 1),
}
project = "DMV_Detection_YOLO"
name = "model0_train_param1_tune"
comet_ml.login(project_name=project)

md = "/home/lab_NiT/huwl/TargetDetection/DMV_Detection_YOLO/model0_train_param1/weights/best.pt"
ds = "/home/lab_NiT/huwl/TargetDetection/cleaned_data/2025-10-30_4-Fold_Cross-val/split_1/split_1_dataset.yaml"
model = YOLO(md)

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data=ds,
    epochs=1024,
    iterations=256,
    optimizer="AdamW",
    space=search_space,
    patience=0,
    plots=False,
    save=False,
    val=True,
)
