#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : kfold.py
# Time       : 2025/10/20 21:09
# Author     : Jago
# Email      : huwl@hku.hk
# Description：https://docs.ultralytics.com/zh/guides/kfold-cross-validation/#k-fold-dataset-split
"""
from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import KFold
import datetime
import shutil
from tqdm import tqdm


ksplit = 4
img_ext = "*.png"
lbl_ext = "*.txt"
# replace with 'path/to/dataset' for your custom data
dataset_path = Path("/home/lab_NiT/huwl/TargetDetection/cleaned_data")
all_images = sorted((dataset_path / "map").rglob(img_ext))
labels = sorted((dataset_path / "label").rglob(lbl_ext))

# your data YAML with data directories and names dictionary
yaml_file = "/home/lab_NiT/huwl/TargetDetection/0_1020.yaml"
with open(yaml_file, encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

index = [label.stem for label in labels]  # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=index)

for label in labels:
    lbl_counter = Counter()

    with open(label) as lf:
        lines = lf.readlines()

    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(line.split(" ", 1)[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

print(labels_df)

random.seed(0)  # for reproducibility
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
kfolds = list(kf.split(labels_df))
folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=index, columns=folds)

for i, (train, val) in enumerate(kfolds, start=1):
    # use .loc with single-step assignment to avoid chained-assignment issues
    folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
    folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()
    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

# KEEP ONLY those that have a corresponding label file.
labeled_stems = set(labels_df.index)
# filter images to those that have labels
images = [img for img in all_images if img.stem in labeled_stems]
skipped = len(all_images) - len(images)
if skipped:
    print(f"[INFO] Skipped {skipped} files that have no labels, i.e. they will not be included in following training or validation.")

# Create the necessary directories and dataset YAML files
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "channels": 1,  # gray images
                "names": classes,
            },
            ds_y,
        )

for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
