import os
import sys
import torch
import torch.nn as nn
import yaml

# Add parent test/ dir so we can import from ImageReward/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_test_dir = os.path.dirname(_this_dir)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from transformers import BertTokenizer


def _init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    # tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    # modify to match the new version of transformer
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids('[ENC]')
    return tokenizer


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0.0)

    def forward(self, x):
        return self.layers(x)


class ImageReward_Model(nn.Module):
    def __init__(self, med_config=None):
        super().__init__()
        if med_config is None:
            med_config = os.path.join(_test_dir, 'ImageReward', 'med_config.json')
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.mlp = MLP(768)
        self.tokenizer = _init_tokenizer()

    def forward(self, batch_data):
        images = batch_data['image']
        prompts = batch_data['prompt']

        text_input = self.tokenizer(
            prompts, padding='max_length',
            truncation=True, max_length=35, return_tensors="pt"
        )
        text_ids = text_input.input_ids.to(images.device)
        text_mask = text_input.attention_mask.to(images.device)

        image_embeds = self.blip.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        text_output = self.blip.text_encoder(
            text_ids,
            attention_mask=text_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True
        )

        last_emb = text_output.last_hidden_state[:, 0, :].float()
        reward = self.mlp(last_emb).squeeze(-1)
        return {'reward_pred': reward}
