import sys
import os

sys.path.append(os.path.abspath("../common"))

import argparse
from pathlib import Path
import numpy as np
import yaml
from functools import partial
from tqdm import tqdm
from functools import partial


import dac_sdc
from iris_utils_v6 import TensorRTEngine


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('experiment_dir', type=Path, help='Path to the experiment directory')
    parser.add_argument('--option', type=str, default='best', help='Model weights to export')
    return parser.parse_args()


def create_callback(engine_path, tqdm_on=False):
    engine = TensorRTEngine(engine_path)
    input_shape = engine.input_shape # (1, 3, h, w)
    if tqdm_on: tbar = tqdm()

    def callback(inputs, tqdm_on=False):
        outputs = dict()
        for img_path, rgb_img in inputs:
            orig_img_shape = engine.preprocess(rgb_img)
            preds = engine.run()
            outputs[img_path.name] = engine.postprocess(preds, orig_img_shape)
            if tqdm_on: tbar.update(1)
        return outputs
    
    return callback, input_shape


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_args = yaml.safe_load(f)
    return yaml_args


def main(args):
    train_path = args.experiment_dir / 'train'
    engine_path = train_path /  'weights' / f'{args.option}.trt'
    experiment_args = read_yaml(train_path / 'args.yaml')
    data_args = read_yaml(experiment_args['data'])
    image_dir = Path(data_args['path']) / data_args['val'] / 'images'
    result_dir = args.experiment_dir / 'results' / args.option
    result_dir.mkdir(parents=True, exist_ok=True)

    dac_sdc.BATCH_SIZE = 1
    dac_sdc.IMG_DIR = image_dir.resolve()
    dac_sdc.RESULT_DIR = result_dir.resolve()
    team = dac_sdc.Team('')
    iris_callback, input_shape = create_callback(engine_path.resolve(), tqdm_on=True)
    _, c, h, w = input_shape
    print("warmup")
    iris_callback(
        [(Path('_.txt'), np.zeros((h, w, c), dtype=np.uint8)) for _ in range(100)],
        tqdm_on=False
    )
    print("done")
    team.run(partial(iris_callback, tqdm_on=True), debug=False)


if __name__ == '__main__':
    main(parse_args())




