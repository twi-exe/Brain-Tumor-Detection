import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd,
    CropForegroundd, EnsureTyped, RandCropByPosNegLabeld, RandAffined, Rand3DElasticd,
    RandBiasFieldd, RandHistogramShiftd, RandGaussianNoised, RandCoarseDropoutd, 
    RandSpatialCropd, RandFlipd, RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd, 
    RandRotate90d, MapTransform
)
import numpy as np
import torch
import random
import warnings


# ---- Custom Augmentations ----

class TumorModalityDropout(MapTransform):
    """
    Randomly zero out T1Gd or FLAIR channel to simulate missing tumor-sensitive modalities.
    Assumes keys in data dict are ['image'] and image has concatenated modalities in channel dim.
    """
    def __init__(self, keys, dropout_prob=0.15, t1gd_idx=1, flair_idx=3):  # adjust idx as per your order
        super().__init__(keys)
        self.dropout_prob = dropout_prob
        self.t1gd_idx = t1gd_idx
        self.flair_idx = flair_idx

    def __call__(self, data):
        d = dict(data)
        if random.random() < self.dropout_prob:
            # randomly choose between T1Gd or FLAIR
            which = random.choice([self.t1gd_idx, self.flair_idx])
            img = d[self.keys[0]]
            if isinstance(img, torch.Tensor):
                img = img.clone()
            else:
                img = np.copy(img)
            img[which] = 0
            d[self.keys[0]] = img
        return d


class MorphLabelAug(MapTransform):
    """
    Randomly dilate or erode tumor label mask to simulate annotation variability.
    Only applied to label (segmentation).
    """
    def __init__(self, keys, prob=0.2):
        super().__init__(keys)
        from scipy.ndimage import binary_dilation, binary_erosion
        self.binary_dilation = binary_dilation
        self.binary_erosion = binary_erosion
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if random.random() < self.prob:
            mask = d[self.keys[0]]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            # Randomly choose dilate or erode
            if random.random() < 0.5:
                mask = self.binary_dilation(mask, iterations=1)
            else:
                mask = self.binary_erosion(mask, iterations=1)
            d[self.keys[0]] = mask.astype(np.uint8)
        return d

# ---- Compose Transforms ----

def get_train_transforms(pixdim=(1.0,1.0,1.0), patch_size=(128,128,128)):
    """
    Compose training transforms: preprocessing + augmentation
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], track_meta=False),

        # Patch-based sampling (tumor/non-tumor)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1.0, neg=1.0, num_samples=2,
            image_key="image",
        ),

        # Spatial Augmentations
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.15, 0.15, 0.15),
            mode=("bilinear", "nearest"),
            padding_mode="border"
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 7),
            magnitude_range=(100, 200),
            prob=0.15,
            rotate_range=(0.05, 0.05, 0.05),
            mode=("bilinear", "nearest"),
        ),

        # Intensity Augmentations
        RandBiasFieldd(keys="image", prob=0.3),
        RandHistogramShiftd(keys="image", prob=0.3, num_control_points=5),
        RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.05),

        # Dropout Augmentation
        RandCoarseDropoutd(
            keys="image",
            holes=5, spatial_size=(32,32,32), max_holes=8, prob=0.2
        ),

        # Custom augmentations
        TumorModalityDropout(keys=["image"], dropout_prob=0.15, t1gd_idx=1, flair_idx=3),
        MorphLabelAug(keys=["label"], prob=0.2),
    ])


def get_test_val_transforms(pixdim=(1.0,1.0,1.0)):
    """
    Compose validation/test transforms: only preprocessing, no augmentation
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])


def get_inference_transforms(pixdim=(1.0,1.0,1.0)):
    """
    For inference (no label), just preprocess image.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], track_meta=False),
    ])