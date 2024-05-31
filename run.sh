#!/bin/bash

# g++-10 -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) -I"$CONDA_PREFIX"/include/opencv4/ -I"$CONDA_PREFIX"/lib/python3.10/site-packages/numpy/core/include/ -L"$CONDA_PREFIX"/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs

# CACHE_DIR=/tmp OPENCV_IO_ENABLE_OPENEXR=1 \
#     python src/stylegan3/gen_images.py \
#     --network checkpoints/k00133t_Ours_FID14.6@28.9M_network-snapshot-002343.pkl \
#     --normalize-azimuth=True \
#     --seeds='elevations+1000' \
#     --outdir=out \
#     --azimuth=180 --elevations=10,70


# Constants
CACHE_DIR=datasets/skymangler_skygan_cache/cache

# CACHE_DIR=$CACHE_DIR DNNLIB_CACHE_DIR=$CACHE_DIR python src/stylegan3/calc_metrics.py \
#     --metrics=fid50k_full --gpus=1 --verbose=True \
#     --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \
#     --network=file:///scratch/iamaq/SkyGAN/skymangler_skygan_cache/cache/downloads/stylegan3-t-ffhq-1024x1024.pkl

#     # --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl
# exit

# Training
CACHE_DIR=$CACHE_DIR DNNLIB_CACHE_DIR=$CACHE_DIR python src/stylegan3/train.py \
    --data=datasets/skymangler_skygan_cache/envmap_skylatlong/export_TRAIN.csv \
    --resolution=256 --gamma=2 \
    --cfg=stylegan3-t --gpus=1 \
    --batch=32 --batch-gpu=4 --tick=1 --snap=1 \
    --outdir=output \
    --metrics=fid50k_full \
    --mirror=0 \
    --aug-ada-xflip=0 \
    --aug-ada-rotate90=0 \
    --aug-ada-xint=0 \
    --aug-ada-scale=0 \
    --aug-ada-rotate=1 \
    --aug-ada-aniso=0 \
    --aug-ada-xfrac=0 \
    --normalize-azimuth=True \
    --use-encoder=True \

# TEMP
# --use-encoder=True \ # NEEDS TO BE TRUE!
# --normalize-azimuth=True \ # NEEDS TO BE TRUE!

# Parameters
# --tick How often to print progress
# --snap How often to save snapshots
# --gpus Number of GPUs to use
# --batch-gpu Limit batch size per GPU

# Training
# python train.py
#     --data /projects/SkyGAN/clouds_fisheye/auto_processed/auto_processed_20230405_1727.csv
#     --resolution=256 --gamma=2
#     --cfg=stylegan3-t --gpus=2
#     --batch=32 --batch-gpu=4 --tick=1 --snap=10
#     --outdir=/local/stylegan3-encoder
#     --metrics=fid50k_full
#     --mirror=0
#     --aug-ada-xflip=0
#     --aug-ada-rotate90=0
#     --aug-ada-xint=0
#     --aug-ada-scale=0
#     --aug-ada-rotate=1
#     --aug-ada-aniso=0
#     --aug-ada-xfrac=0
#     --normalize-azimuth=True
#     --use-encoder=True
