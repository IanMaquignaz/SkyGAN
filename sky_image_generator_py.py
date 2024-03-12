'''
A non-jupiter notebook version of the sky_image_generator_py.ipynb file.
'''

# Standard Library
import os
import math

# Machine Vision
import cv2
import numpy as np

# Junk
import pandas as pd
from matplotlib import pyplot as plt


# Custom
import sky_image_generator

coordinate_img = np.array(sky_image_generator.project_cloudplane_coordinates(128)).reshape(3,128,128)
coordinate_img = np.moveaxis(coordinate_img, 0, 2)

layer = coordinate_img[:,:,0]
layer.min(), layer.max()

plt.imshow(coordinate_img)



# load the shooting metadata
data = pd.read_csv("light_cloud_cover.csv")
shoot, photo_index = '2019-06-24_0504_prague_chodska_21_rooftop', '0418' # done2

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

fname = '/projects/SkyGAN/clouds_fisheye/processed/'+shoot+'/1K_EXR/IMG_'+photo_index+'_hdr.exr'
real_img = cv2.imread(fname, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

view_multiplier = 8
plt.imshow(cv2.cvtColor(real_img*view_multiplier, cv2.COLOR_BGR2RGB))


# look up image's elevation and azimuth in the shooting metadata
data_line = data[data['img_fname'] == fname]
assert len(data_line) == 1
elevation, azimuth = data_line['img_elevation'].iloc[0], data_line['img_azimuth'].iloc[0]

# verify (visually) the generated sky image looks similar
model_img = sky_image_generator.generate_image(
    1024, #resolution
    elevation /180*np.pi, # elevation
    math.fmod(360 + 270 - azimuth, 360) /180*np.pi, # azimuth
    100, # visibility (in km)
    0.1 # ground albedo
)
plt.imshow(cv2.cvtColor(model_img/(2**8)*view_multiplier, cv2.COLOR_BGR2RGB))


# masking out irrelevant areas (close to the Sun and outside the projected skydome)
#t_min = 1e-10
#t_min = 0.003
#t_max = 0.04
t_min, t_max = np.percentile(real_img, [30, 99])
print(t_min, t_max)
mask_thresh = cv2.inRange(real_img, (t_min, t_min, t_min), (t_max, t_max, t_max))
mask_thresh = mask_thresh.astype("bool")
plt.imshow(mask_thresh)


# https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv


#real_img_y = cv2.cvtColor(real_img*10000, cv2.COLOR_BGR2YUV)[..., 0]
real_img_y = RGB2YUV(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))[..., 0]
#print(np.percentile(real_img, [0,5,25,50,75,95,100]))
real_img_y = np.clip(real_img_y, *np.percentile(real_img, [5, 95]))
#print(np.percentile(real_img, [0,5,25,50,75,95,100]))
real_img_y = real_img_y - real_img_y.min()
#print(np.percentile(real_img, [0,5,25,50,75,95,100]))
real_img_y = real_img_y / real_img_y.max()
#print(np.percentile(real_img, [0,5,25,50,75,95,100]))
#print(real_img_y.shape, real_img_y.min(), real_img_y.mean(), real_img_y.max())
plt.imshow(real_img_y)
#print(np.percentile(real_img, [0,5,25,50,75,95,100]))


t_min, t_max = np.percentile(real_img_y, [30, 95])
print(t_min, t_max)
mask_thresh = cv2.inRange(real_img_y, t_min, t_max)
mask_thresh = mask_thresh.astype("bool")
plt.imshow(mask_thresh)


mask_name = fname.split('/')[5] + '-' + fname.split('/')[7].replace('.exr', '')
mask_fname_base = 'masks/' + mask_name + '/'
if not os.path.isdir(mask_fname_base):
    os.mkdir(mask_fname_base)


# laod and use a manually-painted mask
mask_thresh_read = cv2.imread(mask_fname_base + 'mask.png') / 255
mask_thresh_read = mask_thresh_read.astype("bool")
mask_thresh_read.mean()
mask_thresh_read = mask_thresh_read[:, :, 0] # use only the "red" channel
mask_thresh_read.shape
mask_thresh_read = mask_thresh_read > 0.5
plt.imshow(mask_thresh_read)
mask_thresh = mask_thresh_read

rgb_mask = np.repeat(mask_thresh[:, :, np.newaxis], 3, axis=2)

plt.imshow(cv2.cvtColor(real_img*rgb_mask*view_multiplier, cv2.COLOR_BGR2RGB))


######################
# Optimize Sky Model #
######################
import scipy.optimize

visibility_min = 20
visibility_max = 131.8

def make_visibility(v): # [0,1] -> [visibility_min, visibility_max]
    return visibility_min + (visibility_max - visibility_min) * v

make_visibility(0.5301181736571103)


def penalize_out_of_range(value, minimum=0, maximum=1):
    if value < minimum:
        return minimum - value
    if value > 1:
        return value - maximum
    return 0

# "exposure": find a multiplier of the generated sky image that minimizes the difference to the real image

def generate_sky_image_exposure(opt_params):
    exposure = opt_params[0]
    return model_img * np.power(2, exposure)

def extra_loss_exposure(opt_params):
    return 0

optimise_exposure = {
    #                  e
    'initial_values': [-8],
    'generate_sky_image': generate_sky_image_exposure,
    'extra_loss': extra_loss_exposure
}

# "exposure, visibility and ground_albedo": find an exposure and model parameters (visibility and ground albedo)

resolution = 1024

def generate_sky_image_exposure_visibility_groundalbedo(opt_params):
    exposure, visibility, ground_albedo = opt_params[0], make_visibility(opt_params[1]), opt_params[2]

    sun_phi = math.fmod(360 + 270 - azimuth, 360) /180*np.pi # azimuth
    sun_theta = elevation /180*np.pi # elevation

    print('generate_sky_image(', [exposure, visibility, ground_albedo], ')')

    model_img = sky_image_generator.generate_image(
        1024, #resolution
        sun_theta, # elevation
        sun_phi, # azimuth
        visibility, # visibility (in km)
        ground_albedo # ground albedo
    )
    return model_img * np.power(2, exposure)

def extra_loss_exposure_visibility_groundalbedo(opt_params):
    exposure, visibility01, ground_albedo = opt_params[0], opt_params[1], opt_params[2]
    extra = 0
    # constrain parameters to their limits parameters (HACK. maybe using a constrained optimizer would be more suitable)
    extra += penalize_out_of_range(visibility01)
    extra += penalize_out_of_range(ground_albedo)

    print('extra_loss', extra)
    return extra

optimise_exposure_visibility_groundalbedo = {
    #                   exposure, visibility, ground_albedo
    #'initial_values': [       -8,        0.5,           0.5], # unused
    'generate_sky_image': generate_sky_image_exposure_visibility_groundalbedo,
    'extra_loss': extra_loss_exposure_visibility_groundalbedo
}


def L1(generated_img, real_img):
    #plt.imshow(cv2.cvtColor(generated_img*5, cv2.COLOR_BGR2RGB))
    multiply = 10
    diff = generated_img - real_img
    cv2.imwrite('last_diff.png', diff * 255 * multiply)
    diff_masked = np.abs(diff)*rgb_mask
    cv2.imwrite('last_diff_masked.png', diff_masked * 255 * multiply)
    return (diff_masked).mean()

compute_error_metric = L1



def loss(opt_params):
    global best_value, best_params
    opt_params_log.append(opt_params)
    print('loss(', opt_params, ')')

    generated_img = optimise_settings['generate_sky_image'](opt_params)
    #print('mean', generated_img.mean())
    assert(generated_img.shape == real_img.shape)

    error_value = compute_error_metric(generated_img, real_img)
    print('error_value', error_value)

    loss_value = optimise_settings['extra_loss'](opt_params) + error_value

    opt_loss_log.append(loss_value)

    # save the parameters of the best iteration
    if loss_value < best_value:
        best_value = loss_value
        best_params = opt_params

    return loss_value


best_value, best_params = float('inf'), None # remember the best encountered parameters, useful in case we leave the minimum
opt_loss_log, opt_params_log = [], []

optimise_settings = optimise_exposure
optim_res = scipy.optimize.minimize(
    loss,
    np.array(optimise_settings['initial_values']),
    options={'eps': 1e-04, 'maxiter': 15#, 'gtol': 1e-07
            }
)
opt_params_log_exposure = opt_params_log
best_params_exposure = best_params
print(optim_res)
assert optim_res.success

# Optimise the visibility distance and ground albedo

best_value, best_params = float('inf'), None # remember the best encountered parameters, useful in case we leave the minimum
opt_loss_log, opt_params_log = [], []

def prepend_opt_params_and_call(prefix, what_to_call):
    def bla(opt_params):
        return what_to_call(np.concatenate((prefix, opt_params), axis=None))
    return bla

optimise_visibility_groundalbedo = {
    #                  visibility, ground_albedo
    'initial_values': [       0.5,           0.5],
    'generate_sky_image': prepend_opt_params_and_call(best_params_exposure, generate_sky_image_exposure_visibility_groundalbedo),
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
assert optim_res.success


# Optimise (fine-tune) all the params at the same time: exposure, visibility distance and ground albedo

best_value, best_params = float('inf'), None # remember the best encountered parameters, useful in case we leave the minimum
opt_loss_log, opt_params_log = [], []

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

#opt_timeshift = None
opt_x = optim_res.x

print(optim_res)


print('best:', best_params)
print('visibility:', make_visibility(best_params[1]), 'km')


#used_params = opt_x
used_params = best_params

with open(mask_fname_base + 'used_params.txt', 'w') as f:
    f.write(repr(used_params.tolist()))
    f.write('\n')
    f.write(repr(opt_loss_log))

# real photograph
#plt.imshow(cv2.cvtColor(real_img*view_multiplier, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor((real_img*view_multiplier)**(1/2.2), cv2.COLOR_BGR2RGB))
#cv2.imwrite(mask_fname_base + 'real_image.png', real_img*view_multiplier*255)
cv2.imwrite(mask_fname_base + 'real_image.png', (real_img*view_multiplier)**(1/2.2)*255)


generate_sky_image=generate_sky_image_exposure_visibility_groundalbedo

# generated sky-dome image
plt.imshow(cv2.cvtColor((generate_sky_image(used_params)*view_multiplier)**(1/2.2), cv2.COLOR_BGR2RGB))
#cv2.imwrite(mask_fname_base + 'generated_image.png', generate_sky_image(used_params)*view_multiplier*255)
cv2.imwrite(mask_fname_base + 'generated_image.png', (generate_sky_image(used_params)*view_multiplier)**(1/2.2)*255)

# difference image
plt.imshow(cv2.cvtColor((generate_sky_image(used_params)-real_img)*view_multiplier, cv2.COLOR_BGR2RGB))

# difference image, masked
plt.imshow(cv2.cvtColor((generate_sky_image(used_params)-real_img)*rgb_mask*view_multiplier, cv2.COLOR_BGR2RGB))
#cv2.imwrite('masked_diff.exr', (generate_sky_image(used_params)-real_img)*rgb_mask)

# The same masked difference image, just "amplified" (simulated ~16x longer exposure)
plt.imshow(cv2.cvtColor((generate_sky_image(used_params)-real_img)*rgb_mask*30*view_multiplier, cv2.COLOR_BGR2RGB))

generated = cv2.cvtColor(generate_sky_image(used_params)*view_multiplier, cv2.COLOR_BGR2RGB)
#generated = generated[:,:,0]
print(generated.shape)
plt.imshow(generated)


######################
# Secondary Channels #
######################
import secondary_channels
secondary_channels.init(resolution)

polar_distance = secondary_channels.polar_distance(secondary_channels.phi, secondary_channels.theta, sun_phi, sun_theta)
plt.imshow(polar_distance)

logdistance_to_clouds = np.log(secondary_channels.distance_to_clouds)
plt.imshow(logdistance_to_clouds)

np.expand_dims(polar_distance, 2).shape
np.expand_dims(logdistance_to_clouds, 2).shape
generated.dtype

# add the secondary channels to the generated clear-sky image
generated_with_secondary_channels = np.concatenate([
    generated,
    np.expand_dims(polar_distance, 2),
    np.expand_dims(logdistance_to_clouds, 2)
], axis=2)


def bla(generated_with_secondary_channels):
    generated_with_secondary_channels = generated_with_secondary_channels[:,:,4]
    print(generated_with_secondary_channels.shape, generated_with_secondary_channels.dtype)
    plt.imshow(generated_with_secondary_channels)
bla(generated_with_secondary_channels)