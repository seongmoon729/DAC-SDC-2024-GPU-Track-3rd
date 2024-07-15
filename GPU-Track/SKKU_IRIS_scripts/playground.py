import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pathlib import Path
from ultralytics import YOLO, settings



settings.update({
    "datasets_dir": str(Path(__file__).parent)
})

# model_cfg = 'yolov8n-seg.yaml'
model_cfg = 'cfg/d0.2_w0.25_c1024.yaml'
# model_cfg = 'cfg/d0.2_w0.25_c1024_mul16_v1.yaml'
# model_cfg = 'cfg/d0.2_w0.25_c1024_mul16_v2.yaml'

data_cfg = '../../data/all/dataset.yaml'
model = YOLO(model_cfg)

model.train(data=data_cfg, epochs=1, batch=32, imgsz=336, rect=False, pretrained=True)

