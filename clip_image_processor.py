import torch.nn.functional as F
import torchvision
import torch
from transformers import CLIPImageProcessor


def preprocess(x: torch.Tensor, clip_image_processor: CLIPImageProcessor):
    # resize
    *b, c, h, w = x.shape
    to_h, to_w = (
        clip_image_processor.crop_size["height"],
        clip_image_processor.crop_size["width"],
    )
    x = torchvision.transforms.Resize((to_h, to_w))(x)

    # normalizes
    image_mean = clip_image_processor.image_mean
    image_std = clip_image_processor.image_std

    x = torchvision.transforms.Normalize(mean=image_mean, std=image_std)(x)

    return x
