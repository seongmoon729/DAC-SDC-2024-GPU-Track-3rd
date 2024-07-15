import time
import numpy as np
import cv2

from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray



try:
    from line_profiler import profile
except:
    def profile(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    

class AvgTimeProfiler:
    def __init__(self):
        self.n = 0
        self.v = 0.
    
    def update(self, t):
        self.v = ((self.v * self.n) + t) / (self.n + 1)
        self.n += 1
    
    @property
    def result(self):
        return self.v
    

mod = SourceModule("""
#include <stdint.h>

__global__ void xywh2xyxy(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dw = x[idx * 4 + 2] / 2.0;
        float dh = x[idx * 4 + 3] / 2.0;
        float x_center = x[idx * 4 + 0];
        float y_center = x[idx * 4 + 1];

        x[idx * 4 + 0] = x_center - dw;  // top left x
        x[idx * 4 + 1] = y_center - dh;  // top left y
        x[idx * 4 + 2] = x_center + dw;  // bottom right x
        x[idx * 4 + 3] = y_center + dh;  // bottom right y
    }
}

__global__ void upscale_masks_nearest(const float* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)iw / ow;
        float scale_y = (float)ih / oh;

        int src_x = min((int)(x * scale_x), iw - 1);
        int src_y = min((int)(y * scale_y), ih - 1);

        int src_index = (c * ih * iw) + (src_y * iw + src_x);
        int dst_index = (c * oh * ow) + (y * ow + x);

        output[dst_index] = input[src_index];
    }
}

__global__ void upscale_masks_bilinear(const float* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)(iw - 1) / (ow - 1);
        float scale_y = (float)(ih - 1) / (oh - 1);

        float src_x = x * scale_x;
        float src_y = y * scale_y;

        int x1 = floor(src_x);
        int y1 = floor(src_y);
        int x2 = min(x1 + 1, iw - 1);
        int y2 = min(y1 + 1, ih - 1);

        float dx = src_x - x1;
        float dy = src_y - y1;

        int index11 = (c * ih * iw) + (y1 * iw + x1);
        int index12 = (c * ih * iw) + (y2 * iw + x1);
        int index21 = (c * ih * iw) + (y1 * iw + x2);
        int index22 = (c * ih * iw) + (y2 * iw + x2);

        float value11 = input[index11];
        float value12 = input[index12];
        float value21 = input[index21];
        float value22 = input[index22];

        float value = (1 - dx) * (1 - dy) * value11 + dx * (1 - dy) * value21 + (1 - dx) * dy * value12 + dx * dy * value22;

        int dst_index = (c * oh * ow) + (y * ow + x);
        output[dst_index] = value;
    }
}


__device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    return p1 + 0.5f * t * (p2 - p0 + t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + t * (3.0f * (p1 - p2) + p3 - p0)));
}

__global__ void upscale_masks_cubic(const float* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)iw / ow;
        float scale_y = (float)ih / oh;

        float src_x = x * scale_x;
        float src_y = y * scale_y;

        int x1 = (int)src_x;
        int y1 = (int)src_y;

        float dx = src_x - x1;
        float dy = src_y - y1;

        float result = 0.0f;

        for (int m = -1; m <= 2; ++m) {
            int xm = min(max(x1 + m, 0), iw - 1);
            float col_val = 0.0f;

            for (int n = -1; n <= 2; ++n) {
                int yn = min(max(y1 + n, 0), ih - 1);
                float val_0 = input[(c * ih * iw) + (max(0, yn - 1) * iw + xm)];
                float val_1 = input[(c * ih * iw) + (yn * iw + xm)];
                float val_2 = input[(c * ih * iw) + (min(ih - 1, yn + 1) * iw + xm)];
                float val_3 = input[(c * ih * iw) + (min(ih - 1, yn + 2) * iw + xm)];

                col_val += cubicInterpolate(val_0, val_1, val_2, val_3, dy);
            }

            result += cubicInterpolate(
                input[(c * ih * iw) + (max(0, y1 - 1) * iw + xm)],
                input[(c * ih * iw) + (y1 * iw + xm)],
                input[(c * ih * iw) + (min(ih - 1, y1 + 1) * iw + xm)],
                input[(c * ih * iw) + (min(ih - 1, y1 + 2) * iw + xm)],
                dx
            );
        }

        int dst_index = (c * oh * ow) + (y * ow + x);
        output[dst_index] = result / 255.0f;
    }
}

__global__ void crop_masks(float* masks, float* boxes, int N, int mh, int mw, float width_ratio, float height_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * mh * mw;
    
    if (idx < total_elements) {
        int n = idx / (mh * mw);
        int i = (idx % (mh * mw)) / mw;
        int j = idx % mw;
        
        float x1 = boxes[n * 4 + 0] * width_ratio;
        float y1 = boxes[n * 4 + 1] * height_ratio;
        float x2 = boxes[n * 4 + 2] * width_ratio;
        float y2 = boxes[n * 4 + 3] * height_ratio;
        
        if (j >= x1 && j < x2 && i >= y1 && i < y2) {
            // Do nothing, keep the original value
        } else {
            masks[n * mh * mw + i * mw + j] = 0;
        }
    }
}

__global__ void threshold_masks(const float* input, uint8_t* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        output[idx] = input[idx] > 0.5f ? 1 : 0;
    }
}

__global__ void resize_and_normalize_nearest(const uint8_t* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)iw / ow;
        float scale_y = (float)ih / oh;

        int src_x = min((int)(x * scale_x), iw - 1);
        int src_y = min((int)(y * scale_y), ih - 1);

        int src_index = (src_y * iw * channels) + (src_x * channels) + c;
        int dst_index = (c * oh * ow) + (y * ow + x);

        output[dst_index] = input[src_index] / 255.0f;
    }
}

__global__ void resize_and_normalize_bilinear(const uint8_t* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)iw / ow;
        float scale_y = (float)ih / oh;

        float src_x = x * scale_x;
        float src_y = y * scale_y;

        int x0 = (int)src_x;
        int x1 = min(x0 + 1, iw - 1);
        int y0 = (int)src_y;
        int y1 = min(y0 + 1, ih - 1);

        float x_diff = src_x - x0;
        float y_diff = src_y - y0;

        int idx_00 = (y0 * iw * channels) + (x0 * channels) + c;
        int idx_01 = (y0 * iw * channels) + (x1 * channels) + c;
        int idx_10 = (y1 * iw * channels) + (x0 * channels) + c;
        int idx_11 = (y1 * iw * channels) + (x1 * channels) + c;

        float val_00 = input[idx_00] / 255.0f;
        float val_01 = input[idx_01] / 255.0f;
        float val_10 = input[idx_10] / 255.0f;
        float val_11 = input[idx_11] / 255.0f;

        float val = (val_00 * (1 - x_diff) * (1 - y_diff)) +
                    (val_01 * x_diff * (1 - y_diff)) +
                    (val_10 * (1 - x_diff) * y_diff) +
                    (val_11 * x_diff * y_diff);

        int dst_idx = (c * oh * ow) + (y * ow + x);
        output[dst_idx] = val;
    }
}

__global__ void resize_and_normalize_area(const uint8_t* input, float* output, int iw, int ih, int ow, int oh, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < ow && y < oh && c < channels) {
        float scale_x = (float)iw / ow;
        float scale_y = (float)ih / oh;

        int x0 = (int)(x * scale_x);
        int x1 = min((int)((x + 1) * scale_x), iw);
        int y0 = (int)(y * scale_y);
        int y1 = min((int)((y + 1) * scale_y), ih);

        float sum = 0.0;
        int count = 0;

        for (int i = y0; i < y1; ++i) {
            for (int j = x0; j < x1; ++j) {
                int src_index = (i * iw * channels) + (j * channels) + c;
                sum += input[src_index];
                count++;
            }
        }

        int dst_idx = (c * oh * ow) + (y * ow + x);
        output[dst_idx] = sum / count / 255.0f;
    }
}

__global__ void resize_and_normalize_area_with_padding(
    const uint8_t* input, float* output, int iw, int ih, int ow, int oh, int new_width, int new_height, int channels, int pad_top, int pad_bottom, int pad_value) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < new_width && y < new_height && c < channels) {
        int dst_idx = (c * new_height * new_width) + (y * new_width + x);

        // Initialize padding
        output[dst_idx] = pad_value / 255.0f;

        // Determine if within the resized image bounds
        if (y >= pad_top && y < pad_top + oh && x < ow) {
            float scale_x = (float)iw / ow;
            float scale_y = (float)ih / oh;

            int x0 = (int)(x * scale_x);
            int x1 = min((int)((x + 1) * scale_x), iw);
            int y0 = (int)((y - pad_top) * scale_y);
            int y1 = min((int)(((y - pad_top) + 1) * scale_y), ih);

            float sum = 0.0;
            int count = 0;

            for (int i = y0; i < y1; ++i) {
                for (int j = x0; j < x1; ++j) {
                    int src_index = (i * iw * channels) + (j * channels) + c;
                    sum += input[src_index];
                    count++;
                }
            }

            output[dst_idx] = sum / count / 255.0f;
        }
    }
} 
""")
xywh2xyxy_kernel = mod.get_function("xywh2xyxy")
upscale_masks_kernel = mod.get_function("upscale_masks_nearest")
crop_masks_kernel = mod.get_function("crop_masks")
threshold_masks_kernel = mod.get_function("threshold_masks")
resize_and_normalize_kernel = mod.get_function("resize_and_normalize_area")
resize_and_normalize_with_padding_kernel = mod.get_function("resize_and_normalize_area_with_padding")

@profile
def resize(img_in, img_out, img_in_h, img_in_w, img_in_c):
    img_out_h, img_out_w = img_out.shape[2:]
    
    block = (16, 16, 1)
    grid = ((img_out_w + block[0] - 1) // block[0], (img_out_h + block[1] - 1) // block[1], img_in_c)
    
    resize_and_normalize_kernel(
        img_in, img_out,
        np.int32(img_in_w), np.int32(img_in_h),
        np.int32(img_out_w), np.int32(img_out_h), np.int32(img_in_c),
        block=block, grid=grid
    )

@profile
def resize_and_padding(img_in, img_out, img_in_h, img_in_w, img_in_c):
    img_out_h, img_out_w = img_out.shape[2:]
    
    r = min(img_out_h / img_in_h, img_out_w / img_in_w)

    new_unpad = (int(round(img_in_w * r)), int(round(img_in_h * r)))
    dw, dh = img_out_w - new_unpad[0], img_out_h - new_unpad[1]

    pad_top = dh // 2
    pad_bottom = dh - pad_top

    block = (16, 16, 1)
    grid = ((img_out_w + block[0] - 1) // block[0], (img_out_h + block[1] - 1) // block[1], img_in_c)

    resize_and_normalize_with_padding_kernel(
        img_in, img_out, 
        np.int32(img_in_w), np.int32(img_in_h),
        np.int32(new_unpad[0]), np.int32(new_unpad[1]),
        np.int32(img_out_w), np.int32(img_out_h),
        np.int32(img_in_c), np.int32(pad_top), np.int32(pad_bottom),
        np.int32(114),
        block=block, grid=grid
    )

@profile
def xywh2xyxy(boxes):
    threads_per_block = 16
    num_elements = boxes.shape[0]
    blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block
    xywh2xyxy_kernel(boxes, np.int32(num_elements), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))
    return boxes

@profile
def upscale_masks(input_arr, output_arr, output_shape):
    channels, ih, iw = input_arr.shape
    oh, ow = output_shape
    
    threads_per_block = (16, 16, 1)
    blocks_per_grid = ((ow + threads_per_block[0] - 1) // threads_per_block[0],
                       (oh + threads_per_block[1] - 1) // threads_per_block[1],
                       channels)
    upscale_masks_kernel(input_arr, output_arr, np.int32(iw), np.int32(ih), np.int32(ow), np.int32(oh), np.int32(channels),block=threads_per_block, grid=blocks_per_grid)
    return output_arr[:channels]


@profile
def crop_masks(masks, boxes, shape):
    n, mh, mw = masks.shape
    ih, iw = shape
    
    width_ratio = mw / iw
    height_ratio = mh / ih
    
    block_size = 256
    grid_size = (n * mh * mw + block_size - 1) // block_size
    crop_masks_kernel(masks, boxes, np.int32(n), np.int32(mh), np.int32(mw),
                     np.float32(width_ratio), np.float32(height_ratio),
                     block=(block_size, 1, 1), grid=(grid_size, 1))
    return masks


@profile
def threshold_masks(masks, boolean_masks):
    num_elements = masks.size
    threads_per_block = 256
    blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block

    threshold_masks_kernel(masks, boolean_masks, np.int32(num_elements),
                           block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))
    n = masks.shape[0]
    return boolean_masks[:n]


@profile
def masks2segments(masks, strategy="largest"):
    segments = []
    for x in masks:
        contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([contour.reshape(-1, 2) for contour in contours])
            elif strategy == "largest":  # select largest segment
                c = np.array(contours[np.array([len(contour) for contour in contours]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        seg = c.astype("float32") # type: ignore
        segments.append(seg)
    return segments


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords

@profile
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

@profile
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords

@profile
def find_last_valid_index(arr):
    seen = set()
    for i, v in enumerate(arr):
        if v in seen:
            return i
        seen.add(v)
    return len(arr)

@profile
def artifacts2output(clss, boxes, segs):
    out_list = []
    for cls, box, seg in zip(clss, boxes, segs):
        cls = int(cls)
        if cls < 7:
            box = box.tolist()
            out_list.append({
                "type": cls + 1,
                "x": box[0],
                "y": box[1],
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "segmentation": [],
            })
        else:
            seg = seg.flatten().tolist()
            if seg:
                out_list.append({
                    "type": cls + 1,
                    "x": -1,
                    "y": -1,
                    "width": -1,
                    "height": -1,
                    "segmentation": [seg],
                })
            else:
                pass
    return out_list


class TensorRTEngine:
    def __init__(self, engine_path):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, '') # For TensorRT plugins
        
        main_engine_path = Path(engine_path)
        self.main_engine, self.main_context = self.load_and_create_context(main_engine_path)
        post_engine_path = main_engine_path.parent / (main_engine_path.stem + '_post.trt')
        assert post_engine_path.exists(), f"Post-engine file not found: {post_engine_path}"
        self.post_engine, self.post_context = self.load_and_create_context(post_engine_path)
        
        self.allocate_buffers()

    def load_and_create_context(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context
        
    def allocate_buffers(self):   
        self.main_bindings, self.post_bindings, self.main_outputs = [], [], []
        self.output_shapes = dict()
        self.outputs = dict()
                
        for i in range(self.main_engine.num_bindings):
            name  = self.main_engine.get_binding_name(i)
            shape = self.main_engine.get_binding_shape(i)
            size  = trt.volume(shape)
            dtype = trt.nptype(self.main_engine.get_binding_dtype(i))
            
            gpu_mem = gpuarray.GPUArray(shape, dtype)
            self.main_bindings.append(gpu_mem.ptr)
            
            if name == 'images':
                assert self.main_engine.binding_is_input(i)
                self.input = gpu_mem
                self.input_shape = shape
                self.input_dtype = dtype
                self.input_tmp = cuda.mem_alloc(4096*4096*3)
            elif name == 'output0': # mixes
                self.mixes_binding = gpu_mem.ptr
                self.main_outputs.append(gpu_mem)
            elif name == 'output1': # protos
                self.protos_binding = gpu_mem.ptr
                self.main_outputs.append(gpu_mem)
            else:
                raise ValueError()
            
        for i in range(self.post_engine.num_bindings):
            name  = self.post_engine.get_binding_name(i)
            shape = tuple(self.post_engine.get_binding_shape(i))
            size  = trt.volume(shape)
            dtype = trt.nptype(self.post_engine.get_binding_dtype(i))
            
            if name == 'mixes':
                assert self.post_engine.binding_is_input(i)
                self.post_bindings.append(self.mixes_binding)
            elif name == 'protos':
                assert self.post_engine.binding_is_input(i)
                self.post_bindings.append(self.protos_binding)
            elif name in ['boxes']:
                gpu_mem = gpuarray.GPUArray(shape, dtype)
                self.post_bindings.append(gpu_mem.ptr)
                self.outputs[name] = gpu_mem
            elif name in ['masks']:
                gpu_mem = gpuarray.GPUArray(shape, dtype)
                self.post_bindings.append(gpu_mem.ptr)
                self.outputs[name] = gpu_mem
                mask_shape = (shape[0], self.input_shape[2], self.input_shape[3])
                self.upsampled_masks = gpuarray.empty(mask_shape, dtype=np.float32)
                self.boolean_masks = gpuarray.empty(mask_shape, dtype=np.uint8)
            elif name in ['indices', 'classes']:
                gpu_mem = cuda.managed_empty(shape, dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
                self.post_bindings.append(int(gpu_mem.base.get_device_pointer()))
                self.outputs[name] = gpu_mem
                self.output_shapes[name] = shape
            else:
                raise ValueError()
    
    @profile
    def preprocess(self, rgb_img):
        ih, iw, c = rgb_img.shape

        cuda.memcpy_htod(self.input_tmp, rgb_img)

#         resize(self.input_tmp, self.input, ih, iw, c)
        resize_and_padding(self.input_tmp, self.input, ih, iw, c)
        return (ih, iw)
    
    @profile
    def postprocess(self, outputs, orig_img_shape):
        target_shape = self.input_shape[2:]
        
        classes = outputs['classes']
        indices = outputs['indices']
        boxes = outputs['boxes']
        masks = outputs['masks']
        
        idx = find_last_valid_index(indices[:, 1])
        classes = classes[:idx, ...]
        boxes = boxes[:idx, ...]
        boxes_host = np.empty(boxes.shape, np.float32)
        stream = cuda.Stream()
        masks = masks[:idx, ...]
        boxes = xywh2xyxy(boxes)
        cuda.memcpy_dtoh_async(boxes_host, boxes.gpudata, stream)

        masks = crop_masks(masks, boxes, target_shape)
        masks = upscale_masks(masks, self.upsampled_masks, target_shape)
        masks = threshold_masks(masks, self.boolean_masks)

        masks = masks.get()
        segs = masks2segments(masks, strategy="concat")

        stream.synchronize()
        
        boxes = scale_boxes(target_shape, boxes_host, orig_img_shape)
        segs = [
            scale_coords(target_shape, x, orig_img_shape)
            for x in segs
        ]
        out_list = artifacts2output(classes, boxes, segs)
        return out_list

    @profile
    def run(self):
        self.main_context.execute_v2(self.main_bindings)
        self.post_context.execute_v2(self.post_bindings)
        return self.outputs


if __name__ == '__main__':
    import sys
    import random
    from tqdm import tqdm
    
    if len(sys.argv) < 2:
        engine_path = "IRIS.trt"
    else:
        engine_path = sys.argv[1]
       
    n = 1000
    seed = 35123
    random.seed(seed)
    img_dir = Path('../images')
    img_paths = sorted(img_dir.glob('*.jpg'))
    random.shuffle(img_paths)
    img_paths = img_paths[:n]
    print('number of images:', len(img_paths))
    print('random seed:', seed)
    
    atp1 = AvgTimeProfiler()
    atp2 = AvgTimeProfiler()
    atp3 = AvgTimeProfiler()
    atp4 = AvgTimeProfiler()
    
    engine = TensorRTEngine(engine_path)
    
    print('warmup')
    [engine.run() for _ in range(100)]
    print('warmup done')
    
    result = dict()
    with tqdm(img_paths, total=len(img_paths)) as tbar:
        for img_path in tbar:
            bgr_img = cv2.imread(str(img_path))
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            t1 = time.time()
            orig_img_shape = engine.preprocess(rgb_img)
            t2 = time.time()
            preds = engine.run()
            t3 = time.time()
            out_list = engine.postprocess(preds, orig_img_shape)
            t4 = time.time()
            atp1.update(t2 - t1)
            atp2.update(t3 - t2)
            atp3.update(t4 - t3)
            tbar.set_postfix(
                pre=atp1.result,
                engine=atp2.result,
                post=atp3.result,
            )
            result[img_path.name] = out_list
            tbar.update(1)