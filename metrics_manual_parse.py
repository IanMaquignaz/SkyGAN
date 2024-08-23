# Standard Library
import os
from os.path import join
from glob import glob

# Machine Learning/Vision
import torch
from torchmetrics import MetricCollection

# Custom
from utils_cv.io.opencv import load_image


ROOT = "output_metrics"
MODEL="00003-stylegan3-t-export_TRAIN-gpus4-batch32-gamma2"
# SUBSET="TEST"
SUBSET="EVALGRID"

PATH = join(ROOT, MODEL, SUBSET)
print(f"PATH={PATH}")

IMAGE_REGEX = "*.png"
IMAGE_PATHS = glob(join(PATH, IMAGE_REGEX))
# print(f"Found {len(IMAGES_PATHS)} images...")
# Remove *_gt.png image paths
IMAGE_PATHS_FAKE = [ p for p in IMAGE_PATHS if 'gt' not in p ]
IMAGE_PATHS_REAL = [ p for p in IMAGE_PATHS if 'gt' in p ]
print(f"Found {len(IMAGE_PATHS)} images...")
assert len(IMAGE_PATHS_REAL) == len(IMAGE_PATHS_REAL)

# # Metrics
# from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment as CLIP_IQA # CLIP-IQA

metrics_cLDR = MetricCollection([
#     MetricCollection([
#         CLIP_IQA(
#             model_name_or_path="openai/clip-vit-large-patch14-336",
#             prompts=('quality',),
#             data_range=1.0
#         )
#     ], postfix='_real'),
#     MetricCollection([
#         CLIP_IQA(
#             model_name_or_path="openai/clip-vit-large-patch14-336",
#             prompts=('quality',), #'real','natural'),
#             data_range=1.0
#         ) # DANGER! Returns dict if more than one prompt provided.
#     ], postfix='_fake'),
])

for p_real, p_fake in zip(IMAGE_PATHS_REAL, IMAGE_PATHS_FAKE):
    img_real_LDR = load_image(p_real)
    img_fake_LDR = load_image(p_fake)

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
    metric_output = {}
    metric_output_cLDR = metrics_cLDR.compute()
    for k in metric_output_cLDR.keys():
        if 'KernelInceptionDistance' in k:
            kid_mean, kid_std = metric_output[k]
            metric_output['KID_mean'] = kid_mean
            metric_output['KID_std'] = kid_std
        elif 'InceptionScore' in k:
            is_mean, is_std = metric_output[k]
            tail = '_fake' if '_fake' in k else '_real'
            metric_output['IS_mean'+tail] = is_mean
            metric_output['IS_std'+tail] = is_std
        elif 'CLIPImageQualityAssessment' in k:
            CLIP_IQA = metric_output[k].mean()
            tail = '_fake' if '_fake' in k else '_real'
            metric_output['CLIP_IQA'+tail] = CLIP_IQA

    with open(join(PATH,"_metrics_manual_.txt"), 'a') as metrics_file:
        for k in metric_output.keys():
            metrics_file.write(f"{k}: {metric_output[k].cpu().item()}\n")
