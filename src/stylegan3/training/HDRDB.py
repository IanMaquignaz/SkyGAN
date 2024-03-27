# Standard Library
import os
from pathlib import Path

# Machine Vision
import cv2
import numpy as np

# Custom
from envmap import EnvironmentMap

# Export
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# DATA_DIR="datasets/skymangler/envmap_skylatlong/"

def load_image(path):
    # load the real image
    print(path)
    if isinstance(path, Path):
        path = path.as_posix()
    try:
        img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading file: {path}")
        raise Exception(f"Error: {e}")
    return img


def load_HDRDB_envmap(path):

    # Load the image
    img = load_image(path)

    if img.shape[0] != img.shape[1]:
        # Assume skylatlong
        e = EnvironmentMap(img, 'skylatlong').convertTo('skyangular')
        img = e.data

    # Done
    return img


# Parameters for OpenCV imwrite
flags_imwrite_EXR_imwrite_float32 = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_FLOAT # float32
]
def save_image(filename, image):
    # save the image
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image.astype(np.float32), flags_imwrite_EXR_imwrite_float32)
