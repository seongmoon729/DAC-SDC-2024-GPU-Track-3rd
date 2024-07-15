#!/bin/bash

# ENV_NAME=dac
# mamba create -n $ENV_NAME -y python=3.10 pip cuda-toolkit ipykernel ipywidgets -c nvidia/label/cuda-12.1.1
# mamba install -n $ENV_NAME -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# mamba run -n $ENV_NAME pip install --force-reinstall opencv-python-headless

ENV_NAME=dac-cu118
mamba create -n $ENV_NAME -y python=3.10 pip cuda-toolkit ipykernel ipywidgets -c nvidia/label/cuda-11.8.0
mamba install -n $ENV_NAME -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
mamba run -n $ENV_NAME pip install scikit-learn ultralytics 
mamba run -n $ENV_NAME pip install --force-reinstall opencv-python-headless