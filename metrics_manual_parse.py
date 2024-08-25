
# Standard Library
import os
from os.path import join
from glob import glob

# Machine Learning/Vision
import torch
from torchmetrics import MetricCollection

# Custom
from utils_cv.io.opencv import load_image, save_image

ROOT = "output_metrics"
MODEL="00003-stylegan3-t-export_TRAIN-gpus4-batch32-gamma2"
SUBSET="TEST"
# SUBSET="EVALGRID"
# SUBSET="EVALGRID_DEMO"


PATH = join(ROOT, MODEL, SUBSET)
print(f"PATH={PATH}")

IMAGE_REGEX = "*.exr"
IMAGE_PATHS = glob(join(PATH, IMAGE_REGEX))
# print(f"Found {len(IMAGES_PATHS)} images...")
# Remove *_gt.png image paths
IMAGE_PATHS_FAKE = [ p for p in IMAGE_PATHS if 'gt' not in p ]
IMAGE_PATHS_REAL = [ p for p in IMAGE_PATHS if 'gt' in p ]
print(f"Found {len(IMAGE_PATHS)} images... Fake({len(IMAGE_PATHS_FAKE)})/Real({len(IMAGE_PATHS_REAL)})")
assert len(IMAGE_PATHS_FAKE) == len(IMAGE_PATHS_REAL)
assert len(IMAGE_PATHS_REAL) > 0

# Metrics
from utils_ml.metrics import DynamicRange as DR # Dynamic Range
from utils_ml.metrics import EarthMoversDistance as EMD # EMD
from utils_ml.metrics import IntegratedIllumination as II # Integrated Illumination
# from src.stylegan3.training.utils import circular_mask

# expects shape in CHW
import numpy as np
def circular_mask(shape):
    assert len(shape) == 3
    x_coords = np.linspace(-1., 1., shape[2])
    y_coords = np.linspace(-1., 1., shape[1])
    x,y = np.meshgrid(x_coords, y_coords)

    # circular black/white mask for each image
    mask = np.ones(shape, dtype=np.uint8)
    mask[:, (x*x+y*y) > 1.0] = 0
    return mask

img = load_image(IMAGE_PATHS_REAL[0])
shape = img.shape[0]
print(f"Image shape: {img.shape} == {shape}")
mask = torch.from_numpy(circular_mask(img.shape[::-1]))
print("Mask shape:", mask.shape)

envmap_maskValid = mask[0].bool()
envmap_maskBorder = torch.logical_not(envmap_maskValid)
# save_image('mask_valid.exr', mask2D_valid)
# save_image('mask_border.exr', mask2D_border)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    device = torch.cuda.current_device()
    print("Using CUDA:", device)
metrics_HDR = MetricCollection(
    [
        MetricCollection(
            [
                DR(result_unit='EV', result_format='real', mask=envmap_maskValid).to(device),
                II(result_format='real', shape=shape, mask=envmap_maskValid).to(device),
            ], prefix='real/',
        ),
        MetricCollection(
            [
                DR(result_unit='EV', result_format='fake', mask=envmap_maskValid).to(device),
                II(result_format='fake', shape=shape, mask=envmap_maskValid).to(device),
            ], prefix='fake/',
        ),
        EMD(bins=1000, range=[0,20000], mask=envmap_maskBorder).to(device),
    ],
    prefix='_HDR/',
)


from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment as CLIP_IQA # CLIP-IQA
metrics_cLDR = MetricCollection([
    MetricCollection([
        CLIP_IQA(
            model_name_or_path="openai/clip-vit-large-patch14-336",
            prompts=('quality',),
            data_range=1.0
        ).to(device)
    ], postfix='_real'),
    MetricCollection([
        CLIP_IQA(
            model_name_or_path="openai/clip-vit-large-patch14-336",
            prompts=('quality',), #'real','natural'),
            data_range=1.0
        ).to(device) # DANGER! Returns dict if more than one prompt provided.
    ], postfix='_fake'),
])

def gamma(x, b=2.2):
    x = x.pow(1/b)
    return x.clamp(0,1)

def to_LDR(img, drange=(0,1)):
    # See training_loop.save_image_grid
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) / (hi - lo) # fix range
    # img = np.rint(
    #     training.training_loop.linear2srgb(img)*255
    # ).clip(0, 255).astype(np.uint8) # to LDR
    img = linear2srgb(img).clip(0, 1).astype(np.float32) # to LDR
    return img

def linear2srgb(x):
    # adapted to numpy from http://www.cyril-richon.com/blog/2019/1/23/python-srgb-to-linear-linear-to-srgb, based on https://stackoverflow.com/questions/34472375/linear-to-srgb-conversion
    return np.where(x > 0.0031308, 1.055 * (x**(1.0 / 2.4)) - 0.055, 12.92 * x)

from utils_transforms import ToTensor
toTensor = ToTensor()
from tqdm import tqdm
count = 0
# IMAGE_PATHS_REAL = IMAGE_PATHS_REAL[:2]
# IMAGE_PATHS_FAKE = IMAGE_PATHS_FAKE[:2]
for p_real, p_fake in tqdm(zip(IMAGE_PATHS_REAL, IMAGE_PATHS_FAKE), total=len(IMAGE_PATHS_REAL)):
    img_real = toTensor(load_image(p_real))['image'].unsqueeze(0).to(device)
    img_fake = toTensor(load_image(p_fake))['image'].unsqueeze(0).to(device)
    # print(img_real.min(), img_real.max(), img_real.dtype)
    # print(img_fake.min(), img_fake.max(), img_fake.dtype)

    # Metrics HDR
    metrics_HDR.update(img_fake,img_real)

    img_real_LDR = toTensor(
        to_LDR(load_image(p_real))
    )['image'].unsqueeze(0).to(device)
    img_fake_LDR = toTensor(
        to_LDR(load_image(p_fake))
    )['image'].unsqueeze(0).to(device)
    # print(img_real_LDR.min(), img_real_LDR.max(), img_real_LDR.dtype)
    # print(img_fake_LDR.min(), img_fake_LDR.max(), img_fake_LDR.dtype)

    # count +=1
    # save_image(join(ROOT,f'img_real_ldr_{count}.png'), img_real_LDR.clone().squeeze()*255, PNG=True)
    # save_image(join(ROOT,f'img_fake_ldr_{count}.png'), img_fake_LDR.clone().squeeze()*255, PNG=True)

    # Metrics cLDR
    for k in metrics_cLDR.keys():
        if 'KernelInceptionDistance' in k  or 'FrechetInceptionDistance' in k:
            metrics_cLDR[k].update(img_real_LDR.detach().clone(), real=True)
            metrics_cLDR[k].update(img_fake_LDR.detach().clone(), real=False)
        elif 'InceptionScore' in k or 'CLIPImageQualityAssessment' in k:
            if '_real' in k:
                metrics_cLDR[k].update(img_real_LDR.detach().clone())
            else:
                metrics_cLDR[k].update(img_fake_LDR.detach().clone())
        elif 'CLIPScore' in k :
            raise NotImplementedError
            # if '_real' in k:
            #     metrics[k].update(real.detach().clone(), cliptext)
            # else:
            #     metrics[k].update(fake.detach().clone(), cliptext)
        elif 'CLIPImageQualityAssessment' in k :
            if 'real' in k:
                metrics_cLDR[k].update(img_real_LDR.detach().clone())
            else:
                metrics_cLDR[k].update(img_fake_LDR.detach().clone())
        else:
            # SSIM, MS-SSIM,
            metrics_cLDR[k].update(img_fake_LDR.detach().clone(), img_real_LDR.detach().clone())


#####################
## Compute Metrics ##
#####################
metric_output = metrics_HDR.compute()
for k in metrics_HDR.keys():
    if 'EarthMoversDistance' in k:
        from torchvision.utils import save_image as tf_save_image
        hist_error_sum = metrics_HDR['EarthMoversDistance'].plot_cummulative_histogram_error()
        tf_save_image(hist_error_sum, join(PATH,"_hist_error_sum_.png"))

metric_output_cLDR = metrics_cLDR.compute()
for k in metric_output_cLDR.keys():
    if 'KernelInceptionDistance' in k:
        kid_mean, kid_std = metric_output_cLDR[k]
        metric_output['KID_mean'] = kid_mean
        metric_output['KID_std'] = kid_std
    elif 'InceptionScore' in k:
        is_mean, is_std = metric_output_cLDR[k]
        tail = '_fake' if '_fake' in k else '_real'
        metric_output['IS_mean'+tail] = is_mean
        metric_output['IS_std'+tail] = is_std
    elif 'CLIPImageQualityAssessment' in k:
        CLIP_IQA = metric_output_cLDR[k].mean()
        tail = '_fake' if '_fake' in k else '_real'
        metric_output['CLIP_IQA'+tail] = CLIP_IQA

with open(join(PATH,"_metrics_manual_.txt"), 'w') as metrics_file:
    for k in metric_output.keys():
        line = f"{k}: {metric_output[k].cpu().item()}"
        print(line)
        metrics_file.write(line+"\n")
