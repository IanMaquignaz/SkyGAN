# Standard Library
import os
import math
import datetime as dt
from pathlib import Path

# Machine Vision
import cv2
import pandas as pd
import numpy as np

# Custom
from envmap import EnvironmentMap

# Export
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def load_image(path):
    # load the real image
    path = '/home/iamaq/storage/skymangler/envmap_skylatlong' / Path(path)
    if isinstance(path, Path):
        path = path.as_posix()
    real_img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    return real_img


def load_HDRDB_envmap(path):

    # Load the image
    real_img = load_image(path)

    if real_img.shape[0] != real_img.shape[1]:
        # Assume skylatlong
        e = EnvironmentMap(real_img, 'latlong').convertTo('skyangular')
        real_img = e.data

    # Done
    return real_img


# Parameters for OpenCV imwrite
flags_imwrite_EXR_imwrite_float32 = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_FLOAT # float32
]
def save_image(path, image):
    # save the image
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)

    if not isinstance(path, Path):
        path = Path(path)
    filename = (path.parent / (path.stem + '.exr')).as_posix()
    cv2.imwrite(filename, image.astype(np.float32), flags_imwrite_EXR_imwrite_float32)
