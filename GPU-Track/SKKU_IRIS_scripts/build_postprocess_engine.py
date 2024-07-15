import math
import argparse
from pathlib import Path
import tensorrt as trt
import numpy as np


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('engine', type=Path, help='TensorRT engine file path for post-process')
    parser.add_argument('--num-classes', '-nc', type=int, default=10, help='Number of classes')
    parser.add_argument('--max-dets', '-max', type=int, default=50, help='Maximum number of detections')
    parser.add_argument('--iou-thres', '-iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--conf-thres', '-conf', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--workspace-size', '-work', type=int, default=16, help='Workspace size in megabytes')
    return parser.parse_args()


def get_network_output_shape(engine_path):
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger,'') # For TensorRT plugins
    with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    output_shapes = dict()
    for i in range(1, engine.num_bindings):
        shape = engine.get_binding_shape(i)
        if len(shape) == 3:
            output_shapes['mixes'] = shape
        elif len(shape) == 4:
            output_shapes['protos'] = shape
        else:
            raise Exception()
    return output_shapes


def squeeze_first(network, tensor, name):
    shuffle_layer = network.add_shuffle(tensor)
    shuffle_layer.reshape_dims = tensor.shape[1:]
    shuffle_layer.name = name
    return shuffle_layer.get_output(0)


def squeeze_last(network, tensor, name):
    shuffle_layer = network.add_shuffle(tensor)
    shuffle_layer.reshape_dims = tensor.shape[:-1]
    shuffle_layer.name = name
    return shuffle_layer.get_output(0)
    


def apply_EfficientNMS_ONNX_plugin(network, boxes, masks_in, scores, max_dets, iou_thres, conf_thres):
    plugin_creator = trt.get_plugin_registry().get_plugin_creator("EfficientNMS_ONNX_TRT", "1", "")
    if not plugin_creator:
        raise RuntimeError("EfficientNMS_ONNX_TRT plugin not found.")
    
    plugin_fields = [
        trt.PluginField("max_output_boxes_per_class", np.array([max_dets], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("center_point_box", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    nms_plugin = plugin_creator.create_plugin(name="EfficientNMSONNXLayer", field_collection=plugin_field_collection)
    
    # Input shapes:
    #   boxes: [batch, num_candidates, 4]
    #   scores: [batch, num_candidates, num_classes]
    nms_layer = network.add_plugin_v2([boxes, scores], nms_plugin)
    nms_layer.name = "EfficientNMSONNXLayer"

    ind_cls = nms_layer.get_output(0)
    classes = network.add_slice(ind_cls, [0, 1], [max_dets, 1], [1, 1]).get_output(0)
    indices = network.add_slice(ind_cls, [0, 0], [max_dets, 2], [1, 2]).get_output(0)
    
    bbox_gather_layer = network.add_gather_v2(boxes, indices, mode=trt.GatherMode.ND)
    mask_gather_layer = network.add_gather_v2(masks_in, indices, mode=trt.GatherMode.ND)
    
    boxes = bbox_gather_layer.get_output(0)
    masks_in = mask_gather_layer.get_output(0)
    return indices, boxes, masks_in, classes


def build_postprocess_engine(network_engine_path, num_classes, max_dets, iou_thres, conf_thres, workspace_size):
    # Create a builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set build configurations
#     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    config.max_workspace_size = 1 << (20 + int(math.log(workspace_size, 2))) # default: 16MB
    
    output_shapes = get_network_output_shape(network_engine_path)
    mixes = network.add_input(name="mixes", dtype=trt.float32, shape=output_shapes['mixes'])
    protos = network.add_input(name="protos", dtype=trt.float32, shape=output_shapes['protos'])
    
    b, c, mh, mw = protos.shape
    # Reshape for matrix multiplication
    proto_shuffle_layer = network.add_shuffle(protos)
    proto_shuffle_layer.first_transpose = (1, 2, 3, 0)
    proto_shuffle_layer.reshape_dims = (c, -1)
    protos = proto_shuffle_layer.get_output(0)

    # [1, 4+num_classes+num_features, num_candidates] -> [1, num_candidates, 4+num_classes+num_features]
    mixes_shuffle_layer = network.add_shuffle(mixes)
    mixes_shuffle_layer.first_transpose = (0, 2, 1)
    mixes_shuffle_layer.name = "TransposeMixes"
    mixes = mixes_shuffle_layer.get_output(0)

    num_candidates = mixes.shape[1]

    # Args for add_slice: (src, start_idices, sizes, strids)
    boxes_layer = network.add_slice(
        mixes, [0, 0, 0], [b, num_candidates, 4], [1, 1, 1]
    )
    boxes_layer.name = "boxesLayer"
    boxes_layer.precision = trt.float32
    boxes_layer.set_output_type(0, trt.float32)
    boxes = boxes_layer.get_output(0)
    
    scores_layer = network.add_slice(
        mixes, [0, 0, 4], [b, num_candidates, num_classes], [1, 1, 1]
    )
    scores_layer.name = "ScoresLayer"  
    scores_layer.precision = trt.float32
    scores_layer.set_output_type(0, trt.float32)
    scores = scores_layer.get_output(0)
    
    masks_in_layer = network.add_slice(
        mixes, [0, 0, 4 + num_classes], [b, num_candidates, c], [1, 1, 1]
    )
    masks_in_layer.name = "MasksInLayer"
    masks_in_layer.precision = trt.float32
    masks_in_layer.set_output_type(0, trt.float32)
    masks_in = masks_in_layer.get_output(0)
    
    indices, boxes, masks_in, classes = apply_EfficientNMS_ONNX_plugin(
        network, boxes, masks_in, scores,
        max_dets=max_dets, iou_thres=iou_thres, conf_thres=conf_thres
    )

    matmul_layer = network.add_matrix_multiply(
        masks_in, trt.MatrixOperation.NONE, protos, trt.MatrixOperation.NONE
    )
    matmul_layer.precision = trt.float32
    matmul_layer.set_output_type(0, trt.float32)
    matmul_layer.name = "MaskMatMulLayer"
    masks = matmul_layer.get_output(0)
    
    masks_shuffle_layer = network.add_shuffle(masks)
    masks_shuffle_layer.reshape_dims = (-1, mh, mw) # exclude batch dim.
    masks = masks_shuffle_layer.get_output(0)
    
    sigmoid_layer = network.add_activation(masks, type=trt.ActivationType.SIGMOID)
    sigmoid_layer.precision = trt.float32
    sigmoid_layer.set_output_type(0, trt.float32)
    sigmoid_layer.name = "MaskSigmoidLayer"
    masks = sigmoid_layer.get_output(0)
    
    indices.name = 'indices'
    boxes.name = 'boxes'
    masks.name = 'masks'
    classes.name = 'classes'
    
    network.mark_output(indices)
    network.mark_output(boxes)
    network.mark_output(masks)
    network.mark_output(classes)

    engine = builder.build_engine(network, config)

    # Save the engine to file
    postprocess_engine_path = network_engine_path.parent / (network_engine_path.stem + '_post.trt')
    with open(postprocess_engine_path, "wb") as f:
        f.write(engine.serialize())
        

if __name__ == '__main__':
    args = parse_args()
    build_postprocess_engine(
        args.engine,
        args.num_classes,
        args.max_dets,
        args.iou_thres,
        args.conf_thres,
        args.workspace_size
    )
