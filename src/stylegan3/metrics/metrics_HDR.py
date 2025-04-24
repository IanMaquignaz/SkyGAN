# Standard Library
from enum import Enum

# Machine Learning
import torch

# Metrics
import torch.nn.functional as F
from torchmetrics import MetricCollection

# # Metrics - Basic
# from torchmetrics.regression import MeanAbsoluteError as L1 # L1
# from torchmetrics.regression import MeanSquaredError as L2 # L2

# # Metrics - Image
# from torchmetrics.image.inception import InceptionScore as IS # IS
# from torchmetrics.image.kid import KernelInceptionDistance as KID # KID
# from torchmetrics.image.fid import FrechetInceptionDistance as FID # FID
# from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance as mFID # mFID
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS # LPIPS
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM # SSIM
# from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM # MS-SSIM
# from torchmetrics.image import PeakSignalNoiseRatio as PSNR # PSNR
# from torchmetrics.image import VisualInformationFidelity as VIF # VIF

# Metrics - Complex
# from torchmetrics.multimodal.clip_score import CLIPScore as CLIP # CLIP
# from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment as CLIP_IQA # CLIP-IQA

# HDRVDP-3
# https://sourceforge.net/projects/hdrvdp/files/hdrvdp/

# Metrics - Custom
# from utils_ml.metrics import Selective
from utils_ml.metrics import DynamicRange as DR # Dynamic Range
from utils_ml.metrics import EarthMoversDistance as EMD # EMD
from utils_ml.metrics import IntegratedIllumination as II # Integrated Illumination

# Dataset config
# Enumerate the discrete values of classes
class ClassLabels(Enum):
    ''' The discrete labels for the dataset'''
    CLOUDS  = 0
    SKYDOME = 1
    SUN  = 2
    BORDER = 3


# METRICS -- TESTING
def get_metrics_test(
        shape,
        envmap_maskBorder, envmap_maskValid,
        device,
        header='test_G',
    ):

    metrics_test_cLDR = MetricCollection(
        [
            # # (CLIP) CLIPScore
            # MetricCollection([ CLIP(model_name_or_path="openai/clip-vit-large-patch14-336") ] , postfix='_real'),
            # MetricCollection([ CLIP(model_name_or_path="openai/clip-vit-large-patch14-336") ] , postfix='_fake'),

            # (FID) Frechet Inception Distance
            # normalize :: If true, expects [0,1]; if false, expects uint8[0,255]
            # MetricCollection([ FID(feature=64, normalize=True) ], postfix='_f64'),
            # MetricCollection([ FID(feature=192, normalize=True) ], postfix='_f192'),
            # MetricCollection([ FID(feature=768, normalize=True) ], postfix='_f768'),
            # MetricCollection([ fid ], postfix='_f2048'),

            # (MiFID) Memorization-Informed Frechet Inception Distance
            # normalize :: If true, expects [0,1]; if false, expects uint8[0,255]
            # MetricCollection([ mFID(feature=64, normalize=True) ], postfix='_f64'),
            # MetricCollection([ mFID(feature=192, normalize=True) ], postfix='_f192'),
            # MetricCollection([ mFID(feature=768, normalize=True) ], postfix='_f768'),
            # MetricCollection([ mFID(feature=2048, normalize=True) ], postfix='_f2048'),
            # # (KID) Kernel Inception Distance
            # # DANGER! Return kid_mean and kid_std
            # KID(
            #     subset_size=5,
            #     normalize=True, # If true, expects [0,1]; if false, expects uint8[0,255]
            # ),

            # (IS) InceptionScore
            # normalize :: If true, expects [0,1]; if false, expects uint8[0,255]
            # DANGER! The mean and standard deviation of the score are returned
            # MetricCollection([ IS(feature=64, normalize=True) ], postfix='_f64_real'),
            # MetricCollection([ IS(feature=192, normalize=True) ], postfix='_f192_real'),
            # MetricCollection([ IS(feature=768, normalize=True) ], postfix='_f768_real'),
            # MetricCollection([ IS(feature=2048, normalize=True) ], postfix='_f2048_real'),

            # MetricCollection([ IS(feature=64, normalize=True) ], postfix='_f64_fake'),
            # MetricCollection([ IS(feature=192, normalize=True) ], postfix='_f192_fake'),
            # MetricCollection([ IS(feature=768, normalize=True) ], postfix='_f768_fake'),
            # MetricCollection([ IS(feature=2048, normalize=True) ], postfix='_f2048_fake'),

            # (CLIP-IQA) CLIP Image Quality Assessment
            # DANGER! Returns 1-value per image! (No reduction...)
            # MetricCollection([
            #     CLIP_IQA(
            #         model_name_or_path="openai/clip-vit-large-patch14-336",
            #         prompts=('quality',),
            #         data_range=1.0
            #     )
            # ], postfix='_real'),
            # MetricCollection([
            #     CLIP_IQA(
            #         model_name_or_path="openai/clip-vit-large-patch14-336",
            #         prompts=('quality',), #'real','natural'),
            #         data_range=1.0
            #     ) # DANGER! Returns dict if more than one prompt provided.
            # ], postfix='_fake'),
        ],
        prefix=header+'_cLDR/',
    )

    metrics_test_HDR = MetricCollection(
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
        prefix=header+'_HDR/',
    )

    return  metrics_test_cLDR, metrics_test_HDR
