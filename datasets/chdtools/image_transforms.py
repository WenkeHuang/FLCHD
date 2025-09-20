import cv2

# To avoid CPU overhead bug raised by albumentations
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

__all__ = ["img_train_aug", "img_eval_aug"]


def img_train_aug(img_size):
    return A.Compose(
        [
            A.ToGray(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RandomResizedCrop(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )


def img_eval_aug(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )
