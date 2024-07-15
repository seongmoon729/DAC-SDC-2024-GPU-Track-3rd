import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import KFold
# from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--ksplit",
        type=int,
        default=5,
        help="Number of splits for k-fold cross-validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=31,
        help="Random state for reproducible results",
    )
    return parser.parse_args()


def generate_kfold_data(dataset_path: Path, ksplit: int = 5, random_state: int = 31):
    dataset_path = dataset_path.resolve()
    labels = sorted(dataset_path.rglob("train/*labels/*.txt"))  # all data in 'labels'
    yaml_file = dataset_path / "dataset.yaml"  # your data YAML with data directories and names dictionary
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    indx = [l.stem for l in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = Counter()
        with open(label, "r") as lf:
            lines = lf.readlines()
        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(" ")[0])] += 1
        labels_df.loc[label.stem] = lbl_counter
    labels_df = labels_df.fillna(0.0)

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=random_state)  # setting random_state for repeatable results
    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio
    print(fold_lbl_distrb)

    supported_extensions = [".jpg", ".jpeg", ".png"]
    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "train" / "images").rglob(f"*{ext}")))

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(dataset_path.parent / f"KFold_state{random_state}")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=False)
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
        dataset_yaml = split_dir / "dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": os.path.relpath(split_dir.resolve(), Path(__file__).parent),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            dst_img = img_to_path / image.name
            dst_lbl = lbl_to_path / label.name

            rel_src_img_path = os.path.relpath(image, dst_img.parent)
            rel_src_lbl_path = os.path.relpath(label, dst_lbl.parent)

            dst_img.symlink_to(rel_src_img_path)
            dst_lbl.symlink_to(rel_src_lbl_path)


def main(args):
    generate_kfold_data(
        args.dataset_path,
        args.ksplit,
        args.random_state
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)