import argparse
from pathlib import Path
from subprocess import run


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('experiment_dir', type=Path, help='Path to the experiment directory')
    parser.add_argument('--workspace-size', type=int, default=16, help='Workspace size in megabytes')
    parser.add_argument('--option', type=str, default='best', help='Model weights to export')
    return parser.parse_args()


def main(args):
    onnx_path = args.experiment_dir / 'train' / 'weights' / f'{args.option}.onnx'
    main_engine = onnx_path.with_suffix('.trt')

    main_engine_build_cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --fp16 --saveEngine={main_engine} --directIO --explicitBatch --buildOnly --workspace={args.workspace_size}"
    post_engine_build_cmd = f"python build_postprocess_engine.py {main_engine}"

    run(main_engine_build_cmd.split(), check=True)
    run(post_engine_build_cmd.split(), check=True)
    
    
if __name__ == '__main__':
    main(parse_args())
