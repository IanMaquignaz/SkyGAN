#!/bin/bash

# g++-10 -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) -I"$CONDA_PREFIX"/include/opencv4/ -I"$CONDA_PREFIX"/lib/python3.10/site-packages/numpy/core/include/ -L"$CONDA_PREFIX"/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs

# CACHE_DIR=/tmp OPENCV_IO_ENABLE_OPENEXR=1 \
#     python src/stylegan3/gen_images.py \
#     --network checkpoints/k00133t_Ours_FID14.6@28.9M_network-snapshot-002343.pkl \
#     --normalize-azimuth=True \
#     --seeds='elevations+1000' \
#     --outdir=out \
#     --azimuth=180 --elevations=10,70


# Training
python src/stylegan3/train.py \
    --data=datasets/skymangler/envmap_skygan_cache/envmap_skylatlong/export_TRAIN.csv \
    --resolution=256 --gamma=2 \
    --cfg=stylegan3-t --gpus=1 \
    --batch=32 --batch-gpu=16 --tick=1 --snap=1 \
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


# Parameters
# --tick How often to print progress
# --snap How often to save snapshots
# --gpus Number of GPUs to use
# --batch-gpu Limit batch size per GPU

# Training
# python train.py
#     --data /projects/SkyGAN/clouds_fisheye/auto_processed/auto_processed_20230405_1727.csv
#     --resolution=256 --gamma=2
#     --cfg=stylegan3-t --gpus=1
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
