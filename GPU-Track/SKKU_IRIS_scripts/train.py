import os
os.environ['OMP_NUM_THREADS'] = '1'

# import comet_ml
import torch
import argparse
from pathlib import Path
from time import sleep
from ultralytics import YOLO, settings


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('experiment_dir', type=Path, help='Path to the experiment directory')
    parser.add_argument('--data-dir', type=Path, default="../../data/all", help='Path to the training data directory')
    parser.add_argument('--model-cfg', type=str, default='yolov8n-seg.yaml')
    parser.add_argument('--width', type=int, default=448, help='Image width')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--validation' ,'-val', action='store_true', help='Enable validation')
    return parser.parse_args()


def main(args):
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    print(f"Train model on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    sleep(3)

    settings.update({
        "datasets_dir": str(Path(__file__).parent)
    })

    model_cfg = args.model_cfg
    imgsz = args.width
    batch = args.batch
    epochs = args.epochs
    val = args.validation

    data_cfg = args.data_dir / 'dataset.yaml'
    project = str(args.experiment_dir)

    model = YOLO(model_cfg)
    model.train(data=data_cfg, imgsz=imgsz, batch=batch, cos_lr=True, epochs=epochs, rect=False, project=project, device=0, amp=True, cache=False, val=val, exist_ok=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)