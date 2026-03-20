import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import os
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)


def setup_model(hps_version: str = "v2.1", device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    return {
        'model': model,
        'preprocess_val': preprocess_val,
        'tokenizer': tokenizer,
        'device': device,
    }


def score(model_dict, img_path: Union[list, str, Image.Image], prompt: str) -> list:
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']

    # Normalize to list
    if isinstance(img_path, (str, Image.Image)):
        img_path = [img_path]

    result = []
    for one_img in img_path:
        with torch.no_grad():
            if isinstance(one_img, str):
                image = preprocess_val(Image.open(one_img)).unsqueeze(0).to(device=device, non_blocking=True)
            elif isinstance(one_img, Image.Image):
                image = preprocess_val(one_img).unsqueeze(0).to(device=device, non_blocking=True)
            else:
                raise TypeError(f'Unsupported img_path type: {type(one_img)}')
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(float(hps_score[0]))
    return result
