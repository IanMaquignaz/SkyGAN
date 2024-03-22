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

DATA_DIR="data/envmap_skylatlong/"

def load_image(path):
    # load the real image
    path = DATA_DIR / Path(path)
    if isinstance(path, Path):
        path = path.as_posix()
    img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_HDRDB_envmap(path):

    # Load the image
    img = load_image(path)

    if img.shape[0] != img.shape[1]:
        # Assume skylatlong
        e = EnvironmentMap(img, 'latlong').convertTo('skyangular')
        img = e.data

    # Done
    return img


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
