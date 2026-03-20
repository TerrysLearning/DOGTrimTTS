import os
import math
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


preprocess_image = _transform(224)


def load_image_batch(data_folder, p_ids, img_ids, t_id, preprocess=None):
    """Load a batch of images in parallel.

    Args:
        data_folder: root data folder
        p_ids: tensor of prompt ids [B]
        img_ids: tensor of image ids [B]
        t_id: integer timestep (e.g. 28, 21, 14, 7)
        preprocess: image transform (default: preprocess_image)

    Returns:
        images: tensor [B, 3, 224, 224]
    """
    if preprocess is None:
        preprocess = preprocess_image

    def load_fn(i):
        p = p_ids[i].item()
        img = img_ids[i].item()
        path = os.path.join(data_folder, f"p{p:05d}", f"img{img:03d}", f"d{t_id}.png")
        return preprocess(Image.open(path)).unsqueeze(0)

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(load_fn, range(len(p_ids))))
    return torch.cat(results, dim=0)


def get_scheduler(optimizer, total_steps, warmup_ratio=0, warmup_type="linear",
                  scheduler_type="cosine", min_lr_scale=0.0):
    """Custom LR scheduler with warmup + decay."""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            if warmup_type == "linear":
                return float(current_step) / float(max(1, warmup_steps))
            elif warmup_type == "constant":
                return 1.0
            else:
                raise ValueError(f"Unknown warmup type: {warmup_type}")

        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)

        if scheduler_type == "cosine":
            return min_lr_scale + 0.5 * (1.0 - min_lr_scale) * (1 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            return 1.0 - (1.0 - min_lr_scale) * progress
        elif scheduler_type == "exp":
            return (min_lr_scale ** progress)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)
