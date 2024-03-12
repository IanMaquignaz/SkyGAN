#!/bin/bash

# g++-10 -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) -I"$CONDA_PREFIX"/include/opencv4/ -I"$CONDA_PREFIX"/lib/python3.10/site-packages/numpy/core/include/ -L"$CONDA_PREFIX"/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs


CACHE_DIR=/tmp OPENCV_IO_ENABLE_OPENEXR=1 \
    python src/stylegan3/gen_images.py \
    --network checkpoints/k00133t_Ours_FID14.6@28.9M_network-snapshot-002343.pkl \
    --normalize-azimuth=True \
    --seeds='elevations+1000' \
    --outdir=out \
    --azimuth=180 --elevations=10,70