#!/bin/bash

# Constants
CACHE_DIR=datasets/skymangler_skygan_cache/cache

# Clear modules
module --force purge
module restore ENV_SKYGAN_modules
source ENV_SKYGAN/bin/activate

# fid_full matches real & fake count
CACHE_DIR=$CACHE_DIR DNNLIB_CACHE_DIR=$CACHE_DIR python src/stylegan3/calc_metrics.py \
    --data=datasets/skymangler_skygan_cache/envmap_skylatlong/export_EVALGRID_DEMO.csv \
    --network=output_SkyGAN/00003-stylegan3-t-export_TRAIN-gpus4-batch32-gamma2/network-snapshot-003541.pkl \
    --gpus=1 \
    --metrics=export_full \
    --mirror=0 \
    --run_dir=output_metrics \

    # --metrics=export_full,fid_full \
