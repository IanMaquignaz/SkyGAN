'''
A non-jupiter notebook version of the sky_image_generator_py.ipynb file.
'''

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
from hdrtools.sunutils import  sunPosition_pySolar_zenithAzimuth
from hdrtools.sunutils import  sunPosition_pySolar_XYZ

# import sky_image_generator
import PSM_2021 as sky_image_generator


# Globals
global shape, azimuth, elevation, real_img, model_img, rgb_mask
shape = 512
azimuth = 0. # Globally set
elevation = 0. # Globally set
real_img = np.zeros((shape, shape, 3))
model_img = np.zeros((shape, shape, 3))
rgb_mask = np.zeros((shape, shape, 3), dtype=bool)


def make_visibility(
    v,
    visibility_min=20,
    visibility_max=131.8
):
    # [0,1] -> [visibility_min, visibility_max]
    return visibility_min + (visibility_max - visibility_min) * v


def generate_sky_image_exposure(opt_params):
    # "exposure":
    # find a multiplier of the generated sky image
    # that minimizes the difference to the real image
    exposure = opt_params[0]
    return model_img.copy() * np.power(2, exposure)


def extra_loss_exposure(opt_params):
    return 0


def prepend_opt_params_and_call(prefix, what_to_call):
    def bla(opt_params):
        return what_to_call(np.concatenate((prefix, opt_params), axis=None))
    return bla


def extra_loss_exposure_visibility_groundalbedo(opt_params):
    exposure, visibility01, ground_albedo = opt_params[0], opt_params[1], opt_params[2]
    extra = 0

    def penalize_out_of_range(value, minimum=0, maximum=1):
        if value < minimum:
            return minimum - value
        if value > 1:
            return value - maximum
        return 0

    # constrain parameters to their limits parameters (HACK. maybe using a constrained optimizer would be more suitable)
    extra += penalize_out_of_range(visibility01)
    extra += penalize_out_of_range(ground_albedo)
    print('extra_loss', extra)
    return extra


def loss(opt_params):
    global best_value, best_params

    def L1(generated_img, real_img):
        diff = generated_img - real_img
        diff_masked = np.abs(diff)
        diff_masked = diff_masked *rgb_mask
        return diff_masked.mean()

    opt_params_log.append(opt_params)
    print('loss(', opt_params, ')')

    generated_img = optimise_settings['generate_sky_image'](opt_params)

    assert \
        (generated_img.shape[0] == real_img.shape[0]) and \
        (generated_img.shape[1] == real_img.shape[1]) and \
        (generated_img.shape[0] == real_img.shape[1]), \
        f'generated_img.shape({generated_img.shape}) != real_img.shape({real_img.shape})'

    error_value = L1(generated_img, real_img)
    print('error_value', error_value)

    loss_value = optimise_settings['extra_loss'](opt_params) + error_value
    opt_loss_log.append(loss_value)

    # save the parameters of the best iteration
    if loss_value < best_value:
        best_value = loss_value
        best_params = opt_params

    return loss_value


# BT.601 RGB to YUV
# NOT VALID FOR HDR IMAGES! Intended for [0-255] LDR
# See Pg.45-47:  https://www.emva.org/wp-content/uploads/GenICam_PFNC_2_3.pdf
# https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                [0.58700, -0.33126, -0.41869],
                [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv


def generate_clearSky_SkyGAN(opt_params):
    exposure = opt_params[0]
    visibility = make_visibility(opt_params[1])
    ground_albedo = opt_params[2]

    print('generate_sky_image_SkyGAN(', [exposure, visibility, ground_albedo], ')')

    # Generate the clearSky
    model_img = sky_image_generator.generate_image(
        shape, #resolution
        elevation, # elevation
        azimuth, # azimuth
        visibility, # visibility (in km)
        ground_albedo # ground albedo
    )
    return model_img * np.power(2, exposure)


def generate_clearSky_HDRDB(opt_params):
    exposure = opt_params[0]
    visibility = make_visibility(opt_params[1])
    ground_albedo = opt_params[2]

    print('generate_sky_image_HDRDB(', [exposure, visibility, ground_albedo], ')')
    model_img = sky_image_generator.generate_image(
        shape, #resolution
        (np.pi/2)-zenith, # elevation
        azimuth+np.pi, # azimuth
        visibility, # visibility (in km)
        ground_albedo # ground albedo
    )
    return model_img * np.power(2, exposure)


def create_mask():
    # TODO! Never tested!

    # masking out irrelevant areas (close to the Sun and outside the projected skydome)
    t_min, t_max = np.percentile(real_img, [30, 99])
    mask_thresh = cv2.inRange(
        real_img,
        (t_min, t_min, t_min),
        (t_max, t_max, t_max)
    )
    mask_thresh = mask_thresh.astype(bool)
    real_img_y = RGB2YUV(
        real_img
    )[..., 0]
    real_img_y = np.clip(real_img_y, *np.percentile(real_img, [5, 95]))
    real_img_y = real_img_y - real_img_y.min()
    real_img_y = real_img_y / real_img_y.max()

    t_min, t_max = np.percentile(real_img_y, [30, 95])
    mask_thresh = cv2.inRange(real_img_y, t_min, t_max)
    mask_thresh = mask_thresh.astype(bool)

    # load and use a manually-painted mask
    # mask_thresh_read = cv2.imread(mask_fname_base + 'mask.png') / 255
    mask_thresh_read = mask_thresh_read.astype(bool)
    mask_thresh_read = mask_thresh_read[:, :, 0] # use only the "red" channel
    mask_thresh_read = mask_thresh_read > 0.5
    mask_thresh = mask_thresh_read
    rgb_mask = np.repeat(mask_thresh[:, :, np.newaxis], 3, axis=2)
    return rgb_mask


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def load_image(path):
    # load the real image
    if isinstance(path, Path):
        path = path.as_posix()
    real_img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    return real_img


def load_HDRDB_envmap(path, TEST=False):

    # Load the image
    real_img = load_image(path)

    if real_img.shape[0] != real_img.shape[1]:
        # Assume latlong
        e = EnvironmentMap(real_img, 'latlong').convertTo('skyangular')
    else:
        # Assume skyangular
        e = EnvironmentMap(real_img, 'skyangular')
    real_img = e.data

    # Get solar position
    zenith, azimuth = get_coordinate_solar(path)

    # Testing
    if TEST:
        # Add a dot to demonstrate the position of the sun
        # x,y,z = get_coordinate_solar(path, xyz=True)
        x,y,z = azimuthZenith2xyz(azimuth,zenith)
        offset = 5
        c,r = e.world2pixel(x,y,z)
        real_img[
            r-offset:r+offset,
            c-offset:c+offset, :] = 0
        real_img[
            r-offset:r+offset,
            c-offset:c+offset, 0] = 200000
        real_img[r,c] = 0

    # ClearSky
    model_img = sky_image_generator.generate_image(
        real_img.shape[0], #resolution
        (np.pi/2)-zenith, # elevation
        azimuth+np.pi, # azimuth
        100, # visibility (in km)
        0.1 # ground albedo
    )
    # p_img = Path('data_HDRDB') / (path.stem + '_clearSky' + path.suffix)
    # save_image(p_img, model_img)

    # Done
    return real_img, zenith, azimuth, model_img


def azimuthZenith2xyz(
    azimuth,
    zenith=None,
    elevation=None
):
    # Convert Zenith to elevation
    if zenith is not None:
        elevation = (np.pi/2) - zenith
    else:
        assert elevation is not None

    '''
    Please note:
    zenith angle = 90degrees - elevation angle
    azimuth angle = north-based azimuth angles require offset (+90deg) and inversion (*-1) to measure clockwise
    thus, azimuth = (pi/2) - azimuth
    '''
    # Fix azimuth orientation
    azimuth = azimuth - (np.pi/2)

    # Convert to XYZ
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation) # Y axis is up
    z = np.cos(elevation) * np.cos(azimuth)

    # Done
    return x,y,z


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


# HDRDB
# DANGER! Timezone is not CANADA/MONTREAL, EST, DST or UTC-5
# HDRDB uses Atlantic Standard Time (AST) is UTC-4
LOCALITY = {
    'timezone': dt.timezone(dt.timedelta(hours=-4)),
    'latitude': 46.778969,
    'longitude': -71.274914,
    'elevation': 125
}


def get_coordinate_solar(timestamp, xyz=False):
    ''' Get solar position in world coordinates from timestamp '''

    if isinstance(timestamp, Path):
        timestamp = timestamp.as_posix()

    if isinstance(timestamp, str):
        timestamp = dateTime_fromPathStem(timestamp)

    if not xyz:
        zenith, azimuth = sunPosition_pySolar_zenithAzimuth(
            LOCALITY['latitude'],
            LOCALITY['longitude'],
            timestamp,
            LOCALITY['elevation']
        )
        return zenith, azimuth

    x,y,z = sunPosition_pySolar_XYZ(
        LOCALITY['latitude'],
        LOCALITY['longitude'],
        timestamp,
        LOCALITY['elevation']
    )
    return x,y,z


def dateTime_fromPathStem(filePath:str, tz=None):
    ''' Get the datetime from file name '''

    if tz is None:
        tz = LOCALITY['timezone']

    # Get the stem
    stem = Path(filePath).stem.split("_")
    year = int(stem[0][0:4])
    month = int(stem[0][4:6])
    day = int(stem[0][6:8])
    hour = int(stem[1][0:2])
    minute = int(stem[1][2:4])
    second = int(stem[1][4:6])

    return dt.datetime(
        year, month, day,
        hour, minute, second,
        tzinfo=LOCALITY['timezone']
    )


# load the shooting metadata
path_SkyGAN_data = Path("/home/iamaq/storage/SkyGAN")
path_SkyGAN_metadata = path_SkyGAN_data / "auto_processed_20230405_1727.csv"
SkyGAN_data = pd.read_csv(path_SkyGAN_metadata.as_posix())
def load_SkyGAN_envmap(idx:int=None, filename:str=None, TEST=True):
    # look up image's elevation and azimuth in the shooting metadata
    if filename is not None:
        data_line = SkyGAN_data[SkyGAN_data['img_fname'] == filename]
    elif idx is not None:
        data_line = SkyGAN_data.iloc[idx]
    # print(data_line, len(data_line))

    path_img = path_SkyGAN_data / data_line['img_fname']
    elevation_deg = data_line['sun_elevation']
    azimuth_deg   = data_line['sun_azimuth']
    azimuth = math.fmod(360+270 - azimuth_deg, 360) /180*np.pi # azimuth
    elevation = elevation_deg /180*np.pi # elevation

    # Load the image
    real_img = load_image(path_img)
    if TEST:
        e = EnvironmentMap(real_img, 'skyangular')
        azimuth_ = math.fmod(180+270 - azimuth_deg, 360) /180*np.pi # azimuth
        x,y,z = azimuthZenith2xyz(azimuth_, elevation=elevation)

        offset = 5
        c,r = e.world2pixel(x,y,z)
        real_img[
            r-offset:r+offset,
            c-offset:c+offset, :] = 0
        real_img[
            r-offset:r+offset,
            c-offset:c+offset, 0] = 200000
        real_img[r,c] = 0

        p_img = Path('data_SkyGAN') / path_img.name
        save_image(p_img, real_img)

        # ClearSky
        model_img = sky_image_generator.generate_image(
            real_img.shape[0], #resolution
            elevation, # elevation
            azimuth, # azimuth
            100, # visibility (in km)
            0.1 # ground albedo
        )
        p_img = Path('data_SkyGAN') / (path_img.stem + '_clearSky' + path_img.suffix)
        save_image(p_img, model_img)
    return real_img, elevation, azimuth


def test_load_SkyGAN():
    print('---')
    load_SkyGAN_envmap(idx=0)
    print('---')
    load_SkyGAN_envmap(idx=10)
    print('---')
    load_SkyGAN_envmap(idx=100)
    print('---')
    load_SkyGAN_envmap(idx=1000)
    print('---')
    load_SkyGAN_envmap(idx=1001)
    print('---')
    load_SkyGAN_envmap(idx=1002)
    print('---')
    load_SkyGAN_envmap(idx=1003)
    print('---')
    exit()


def test_azimuth():
    count = 13
    for i in np.linspace(0, 360, count):
        model_img = sky_image_generator.generate_image(
            1024, #resolution
            # elevation /180*np.pi, # elevation
            15 /180*np.pi, # elevation
            # math.fmod(360 + 270 - azimuth, 360) /180*np.pi, # azimuth
            math.fmod(360 + 270 - i, 360) /180*np.pi, # azimuth
            100, # visibility (in km)
            0.1 # ground albedo
        )
        p_img = Path('data_SkyGAN') / ('clearSky_azimuth_'+str(int(i)).zfill(3)+'.exr')
        save_image(p_img, model_img)


def test_elevation():
    count = 7
    for i in np.linspace(0, 90, count):
        model_img = sky_image_generator.generate_image(
            1024, #resolution
            # elevation /180*np.pi, # elevation
            i /180*np.pi, # elevation
            # math.fmod(360 + 270 - azimuth, 360) /180*np.pi, # azimuth
            math.fmod(360 + 270 - 0, 360) /180*np.pi, # azimuth
            100, # visibility (in km)
            0.1 # ground albedo
        )
        p_img = Path('data_SkyGAN') / ('clearSky_elevation_'+str(int(i)).zfill(3)+'.exr')
        save_image(p_img, model_img)


######################
# Optimize Sky Model #
######################

import scipy.optimize

paths = [
    '20141005_125753_envmap.exr',
    '20141005_170952_envmap.exr',
    '20160607_125219_envmap.exr',
    '20160610_185510_envmap.exr',
    '20160621_121606_envmap.exr',
    '20141005_154552_envmap.exr',
    '20160607_084600_envmap.exr',
    '20160610_090634_envmap.exr',
    '20160620_103154_envmap.exr',
    '20160621_202836_envmap.exr',
]

for p in paths:
    print(p)

    # Load the image
    p = 'data_HDRDB'/ Path(p)

    real_img, zenith, azimuth, model_img = load_HDRDB_envmap(p)
    shape = real_img.shape[0]

    p_sa = p.parent / (p.stem + '_skyAngular' + p.suffix)
    save_image(p_sa,real_img)

    rgb_mask = np.ones((shape, shape, 3), dtype=bool)


    ############
    # Exposure #
    ############
    best_value, best_params = float('inf'), None
    opt_loss_log, opt_params_log = [], []

    # Set objective (passed by reference via optimize_settings)
    optimise_exposure = {
        #                  e
        'initial_values': [-8],
        'generate_sky_image': generate_sky_image_exposure,
        'extra_loss': extra_loss_exposure
    }
    optimise_settings = optimise_exposure


    optim_res = scipy.optimize.minimize(
        loss,
        np.array(optimise_settings['initial_values']),
        options={
            'eps': 1e-04,
            'maxiter': 30,
            #'gtol': 1e-07,
        }
    )
    opt_params_log_exposure = opt_params_log
    best_params_exposure = best_params
    print(optim_res)
    print('best:', best_params)
    assert optim_res.success

    # Save intermediary result
    model_img = model_img * np.power(2, best_params)
    p_img = Path('data_HDRDB') / (p.stem + '_clearSky' + p.suffix)
    save_image(p_img, model_img)


    ################################
    # visibility and ground albedo #
    ################################
    best_value, best_params = float('inf'), None # remember the best encountered parameters, useful in case we leave the minimum
    opt_loss_log, opt_params_log = [], []

    # Set objective (passed by reference via optimize_settings)
    optimise_visibility_groundalbedo = {
        #                  visibility, ground_albedo
        'initial_values': [       0.5,           0.5],
        'generate_sky_image': prepend_opt_params_and_call(best_params_exposure, generate_clearSky_HDRDB),
        'extra_loss': prepend_opt_params_and_call(best_params_exposure, extra_loss_exposure_visibility_groundalbedo)
    }
    optimise_settings = optimise_visibility_groundalbedo

    optim_res = scipy.optimize.minimize(
        loss,
        np.array(optimise_settings['initial_values']), # replace the initial exposure with the found one
        method='L-BFGS-B',
        bounds=[(0,1), (0,1)],
        options={'eps': 1e-04, 'maxiter': 20, 'pgtol': 1e-07}
    )
    opt_params_log_visibility_groundalbedo = opt_params_log
    best_params_visibility_groundalbedo = best_params
    print(optim_res)
    print('best:', best_params)
    assert optim_res.success

    model_img = optimise_settings['generate_sky_image'](best_params)
    p_img = Path('data_HDRDB') / (p.stem + '_clearSky' + p.suffix)
    save_image(p_img, model_img)


    ##########################################
    # Exposure, visibility and ground albedo #
    ##########################################
    # Optimise (fine-tune) all the params at the same time: exposure, visibility distance and ground albedo
    best_value, best_params = float('inf'), None # remember the best encountered parameters, useful in case we leave the minimum
    opt_loss_log, opt_params_log = [], []

    optimise_exposure_visibility_groundalbedo = {
        #                   exposure, visibility, ground_albedo
        #'initial_values': [       -8,        0.5,           0.5], # unused
        'generate_sky_image': generate_clearSky_HDRDB,
        'extra_loss': extra_loss_exposure_visibility_groundalbedo
    }
    optimise_settings = optimise_exposure_visibility_groundalbedo

    optim_res = scipy.optimize.minimize(
        loss,
        np.array(np.concatenate((best_params_exposure, best_params_visibility_groundalbedo), axis=None)), # replace the initial exposure with the found one
        method='L-BFGS-B',
        bounds=[[None, None], (0,1), (0,1)],
        options={'eps': 1e-04, 'maxiter': 25}
    )

    opt_params_log_exposure_visibility_groundalbedo = opt_params_log
    best_params_exposure_visibility_groundalbedo = best_params


    print(optim_res)
    print('best:', best_params)
    print('visibility:', make_visibility(best_params[1]), 'km')
    model_img = optimise_settings['generate_sky_image'](best_params)
    p_img = Path('data_HDRDB') / (p.stem + '_clearSky_all' + p.suffix)
    save_image(p_img, model_img)


    with open('params.txt', 'a') as f:
        f.write(p.name+' ')
        f.write(repr(best_params.tolist()))
        f.write('\n')
        # f.write(repr(opt_loss_log))


######################
# Secondary Channels #
######################
# import secondary_channels
# secondary_channels.init(resolution)

# polar_distance = secondary_channels.polar_distance(secondary_channels.phi, secondary_channels.theta, sun_phi, sun_theta)
# plt.imshow(polar_distance)

# logdistance_to_clouds = np.log(secondary_channels.distance_to_clouds)
# plt.imshow(logdistance_to_clouds)

# np.expand_dims(polar_distance, 2).shape
# np.expand_dims(logdistance_to_clouds, 2).shape
# generated.dtype

# # add the secondary channels to the generated clear-sky image
# generated_with_secondary_channels = np.concatenate([
#     generated,
#     np.expand_dims(polar_distance, 2),
#     np.expand_dims(logdistance_to_clouds, 2)
# ], axis=2)


# def bla(generated_with_secondary_channels):
#     generated_with_secondary_channels = generated_with_secondary_channels[:,:,4]
#     print(generated_with_secondary_channels.shape, generated_with_secondary_channels.dtype)
#     plt.imshow(generated_with_secondary_channels)
# bla(generated_with_secondary_channels)