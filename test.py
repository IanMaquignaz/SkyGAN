import numpy as np
import math

import sky_image_generator as hw

resolution = 256
elevation=10 # 10,70
azimuth = 180
exposure, visibility, ground_albedo = -9.17947373e+00,  1.00012133e+02,  5.95418177e-03

model_img = hw.generate_image(
    resolution,
    elevation / 180 * np.pi, # elevation
    math.fmod(360 + 270 - azimuth, 360) / 180 * np.pi, # azimuth
    visibility, # visibility (in km)
    ground_albedo # ground albedo
)

print(model_img.shape, model_img.dtype, model_img.min(), model_img.max())

