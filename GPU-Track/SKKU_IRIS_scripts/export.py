import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import math
from time import sleep
from pathlib import Path
from ultralytics import YOLO


def int32divisible(size):
    size = int(size)
    if size % 32 != 0:
        raise argparse.ArgumentTypeError('Size must be a multiple of 32')
    return size


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('experiment_dir', type=Path, help='Path to the experiment directory')
    parser.add_argument('--option', type=str, default='best', help='Model weights to export')
    parser.add_argument('--width', type=int, default=448, help='Width of the input image')
    parser.add_argument('--aspect-ratio', type=float, default=1.78, help='Aspect ratio of the input image')
    parser.add_argument('--stride', type=int, default=32, help='Stride of the model')
    return parser.parse_args()


def main(args):
    ratio = args.width / args.height
    print(f"Export model with input size: {args.width}x{args.height} (aspect ratio: {ratio:.2f})")
    sleep(3)

    height = math.ceil(math.ceil(args.width / args.aspect_ratio) / 32) * 32
    weight_path = args.experiment_dir / 'train' / 'weights' / f'{args.option}.pt'
    model = YOLO(weight_path)
    model.export(format='onnx', simplify=True, imgsz=(height, args.width))
    

if __name__ == '__main__':
    args = parse_args()
    main(args)