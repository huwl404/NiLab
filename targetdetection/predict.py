#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : predict.py
# Time       : 2025/11/8 12:43
# Author     : Jago
# Email      : huwl@hku.hk
# Description：
"""
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./data/md2_pm2_best.pt")

# Define path to the image file
source = "./test/images/mmm001_tile000.png"

# Run inference on the source
results = model(source,
                save_txt=True,
                save_conf=True,
                save_crop=True,
                device="cpu",
                project="whereisproect",
                name="whereisname")  # list of Results objects