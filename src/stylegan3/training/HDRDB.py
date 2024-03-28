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

# Parameters for OpenCV Imread
flags_imread_HDR = (
    cv2.IMREAD_ANYCOLOR | # The image is read in any possible color format
    cv2.IMREAD_ANYDEPTH | # Return 16-bit/32-bit image when the input has the corresponding depth
    cv2.IMREAD_UNCHANGED  # Return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
)
def load_image(path):
    # load the real image
    if isinstance(path, Path):
        path = path.as_posix()
    try:
        img = cv2.imread(path, flags=flags_imread_HDR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading file: {path}")
        raise Exception(f"{e}")
    return img


def load_HDRDB_envmap(path):

    # Load the image
    img = load_image(path)

    if '_envmap.exr' in path:
    # if img.shape[0] != img.shape[1]:
        # Assume skylatlong
        e = EnvironmentMap(img, 'skylatlong').convertTo('skyangular')
        img = e.data.copy()

    # save_image('tmp/'+Path(path).name, img)
    # print("IN << ", path)

    # Done
    return img


# Parameters for OpenCV imwrite
flags_imwrite_EXR_imwrite_float32 = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_FLOAT # float32
]
def save_image(path, img):
    # save the image
    # print("OUT >> ", path)
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img, flags_imwrite_EXR_imwrite_float32)
