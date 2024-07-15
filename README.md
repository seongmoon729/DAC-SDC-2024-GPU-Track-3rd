# DAC-SDC-2024 GPU Track 3rd Place

This repository contains the code for the 3rd place solution in the DAC-SDC-2024 GPU Track.

## Functionality
You can find everything you need in `GPU-Track/SKKU_IRIS_scripts`

## GPU Server: Training & Model export

### **Prerequisites**
This project heavily based on [Ultralytics](https://github.com/ultralytics/ultralytics).
Our training and export codes are build on it. You need to install ultralytics
```bash
pip install ultralytics
```
or just use shell script file `install.sh` for conda environment settings.
```bash
bash install.sh
```

### generate_ultralytics_data.py
This file generate the instance segmentation data as the form of `ultralytics`. In this challenge, our strategy is treating the bounding box as the instance segmentation polygon. Therefore, this script converts all bounding boxes into instance segmentation polygons.

### train.py
This script trains a YOLOv8 segmentation model on a GPU.

### export.py
This script exports a pretrained YOLOv8 segmentation `pytorch` model to `ONNX` model


## Jetson Nano: Build engines & Inference

### **iris_utils_v6.py**
This file contains all the pre-/post-processing and TensorRT engine execution behaviors.

- **Pre-process:**
  - Resize the source image to fit the model input size.
  - Convert data types.
  - Copy data from host (CPU) to device (CUDA).

- **Post-process:**
  - Execute non-maximum suppression and operations for mask generation using the TensorRT engine (`IRIS_post.trt`).
  - Crop masks.
  - Upscale masks.
  - Apply thresholds to masks.
  - Rescale masks and bounding boxes.

- **TensorRTEngine:**
  - Run the neural network using the TensorRT engine (`IRIS.trt`).

### **build_engines.py**
This script generate TensorRT engines (`*.trt`, `*_post.trt`) based on `ONNX` model file.

### **IRIS.trt**
This is the TensorRT engine file for the neural network model, based on a custom YOLOv8n-seg model.

### **IRIS_post.trt**
This TensorRT engine file implements parts of the post-processing, including:
  - Non-maximum suppression.
  - Sigmoid activation and matrix multiplication for mask generation.

### Usage

To run the inference code on Jetson Nano, follow these steps:

1. **Prerequisites:**
   - Ensure you have the required dependencies installed, including TensorRT, pycuda, numpy, opencv-python, and tqdm.
   - We recommend you to use [JetPack SDK 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461)
   - You can install rest of them using pip:
     ```sh
     pip install pycuda tqdm
     ```

2. **Setup:**
   - Place your images in a directory named `images` located in the parent directory of the project. Ensure the images are in `.jpg` format.

3. **Running the Code:**
   - You can run the script by providing the path to the TensorRT engine file (`IRIS.trt`). If no engine file path is provided, it defaults to `IRIS.trt` in the current directory.
     ```sh
     python iris_utils_v6.py [path_to_engine_file]
     ```

   - Example:
     ```sh
     python iris_utils_v6.py IRIS.trt
     ```
      or you can run the script using `dac_sdc.ipynb` notebook

4. **Output:**
   - The script processes the images and generates a result dictionary containing the processed output for each image. The processing includes pre-processing the images, running the TensorRT engine, and post-processing the outputs.

5. **Profiling:**
   - The script includes time profiling for different stages (pre-processing, engine execution, and post-processing) to help analyze performance.
   - For this functionality, you need to install `line_profiler`
   ```sh
    pip install line_profiler
   ```
   - Then you can run the script with an environment variable `LINE_PROFILE=1` 
   ```sh
    LINE_PROFILE=1 python iris_utils_v6.py IRIS.trt
   ```

