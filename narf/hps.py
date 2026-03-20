import os
import sys
import torch
import torch.nn as nn

_this_dir = os.path.dirname(os.path.abspath(__file__))
_rewards_dir = os.path.join(os.path.dirname(_this_dir), 'rewards')
if _rewards_dir not in sys.path:
    sys.path.insert(0, _rewards_dir)

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPS_Model(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model, _, _ = create_model_and_transforms(
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
        self.processor = get_tokenizer('ViT-H-14')

    def forward(self, batch_data):
        images = batch_data['image']
        text_input = self.processor(batch_data['prompt']).to(images.device)

        outputs = self.model(images, text_input)
        image_features = outputs['image_features']
        text_features = outputs['text_features']

        hps_scores = torch.diagonal(image_features @ text_features.T)
        return {'reward_pred': hps_scores}
