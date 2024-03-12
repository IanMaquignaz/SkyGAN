# Dockerfile for running StyleGAN3/SkyGAN

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# StyleGAN3
FROM nvcr.io/nvidia/pytorch:21.08-py3
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 pandas
WORKDIR /workspace

# Update the box (refresh apt-get)
# Packages can be messed up, but using '--fix-missing' results in a full docker rebuild
# This uses '--fix-missing' only if something fails.
RUN apt-get update -y --fix-missing
RUN apt-get update -y && if [ $? -ne 0 ] ; then apt-get update -y --fix-missing ; fi ;

# SkyGAN
RUN apt install g++-10 -y
RUN pip install imgui glfw pyopengl


# Creates a non-root user with an explicit UID
ARG USER_NAME="toor"
ARG USER_ID=5678
ARG GROUP_ID=8765
RUN groupadd -g ${GROUP_ID} docker 
RUN useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
RUN echo "${USER_NAME}:toor" |  chpasswd 
USER $USER_ID:${GROUP_ID}

COPY src/stylegan3/environment.yml environment.yml
RUN conda update -n base conda
RUN conda env create -f environment.yml
RUN conda init bash
RUN conda activate stylegan3 && g++-10 -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) -I"$CONDA_PREFIX"/include/opencv4/ -I"$CONDA_PREFIX"/lib/python3.8/site-packages/numpy/core/include/ -L"$CONDA_PREFIX"/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs
