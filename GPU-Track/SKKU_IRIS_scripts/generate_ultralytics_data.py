import os
import json
import yaml
import argparse
from collections import Counter
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


names = {
    0: "motor vehicle",
    1: "non-motor vehicle",
    2: "pedestrian",
    3: "red traffic light",
    4: "yellow traffic light",
    5: "green traffic light",
    6: "off traffic light",
    7: "solid lane line",
    8: "dotted lane line",
    9: "crosswalk",
}

yaml_dict = {
    'names': names,
}

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the source dataset directory",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--num-validation",
        type=int,
        default=2000,
        help="Number of validation images",
    )
    return parser.parse_args()

def main(args):
    src_img_dir = args.dataset_path / 'JPEGImages'
    src_label_dir = args.dataset_path / 'label'

    root_dir = args.output_path
    if root_dir.exists():
        shutil.rmtree(root_dir)
    
    dst_img_dir = root_dir / 'train' / 'images'
    dst_lbl_dir = root_dir / 'train' / 'labels'
    dst_img_dir.mkdir(parents=True, exist_ok=False)
    dst_lbl_dir.mkdir(parents=True, exist_ok=False)

    dst_val_img_dir = root_dir / 'val' / 'images'
    dst_val_lbl_dir = root_dir / 'val' / 'labels'
    dst_val_img_dir.mkdir(parents=True, exist_ok=False)
    dst_val_lbl_dir.mkdir(parents=True, exist_ok=False)

    yaml_dict['path'] = os.path.relpath(root_dir.resolve(), Path(__file__).parent)
    yaml_dict['train'] = 'train'
    yaml_dict['val'] = 'val'
    with open(root_dir / 'dataset.yaml', 'w') as f:
        yaml.safe_dump(yaml_dict, f)

    label_paths = sorted(src_label_dir.iterdir())
    counter = Counter({i: 0 for i in range(10)})
    with tqdm(label_paths) as pbar:
        for i, src_label_path in enumerate(pbar):
            with open(src_label_path) as f:
                label = json.load(f)
            src_img_path = src_img_dir / src_label_path.name.replace('.json', '.jpg')
            img = Image.open(src_img_path)
            img_w, img_h = img.size

            dst_img_path = dst_img_dir / src_img_path.name
            dst_lbl_path = dst_lbl_dir / src_label_path.name.replace('.json', '.txt')
            with open(dst_lbl_path, 'w') as f:
                for la in label:
                    c = la['type'] - 1
                    counter[c] += 1
                    if c < 7:
                        x = la['x']
                        y = la['y']
                        w = la['width']
                        h = la['height']
                        if w <= 0 or h <= 0:
                            continue
                        seg = [x, y, x+w, y, x+w, y+h, x, y+h]
                    else:
                        assert len(la['segmentation']) == 1
                        seg = la['segmentation'][0]
                    seg = [s/img_w if i % 2 == 0 else s/img_h for i, s in enumerate(seg)]
                    f.write(f'{c:<2d} ' + ' '.join(map(str, seg)) + '\n')
            dst_img_path.symlink_to(src_img_path.resolve())
            pbar.set_postfix(count=str(dict(counter)))
            if i < args.num_validation:
                dst_val_img_path = dst_val_img_dir / src_img_path.name
                dst_val_lbl_path = dst_val_lbl_dir / src_label_path.name.replace('.json', '.txt')
                dst_val_img_path.symlink_to(src_img_path.resolve())
                shutil.copyfile(dst_lbl_path, dst_val_lbl_path)

    

    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)