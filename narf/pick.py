import torch
from transformers import CLIPModel, CLIPProcessor


class PickScore_Model(CLIPModel):

    def setup_processor(self):
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, batch_data):
        images = batch_data['image']
        text_input = self.processor(
            text=batch_data['prompt'], padding=True,
            truncation=True, max_length=77, return_tensors="pt"
        )
        text_ids = text_input['input_ids'].to(images.device)
        text_mask = text_input['attention_mask'].to(images.device)

        image_embs = self.get_image_features(pixel_values=images)
        if not isinstance(image_embs, torch.Tensor):
            image_embs = image_embs.pooler_output
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.get_text_features(input_ids=text_ids, attention_mask=text_mask)
        if not isinstance(text_embs, torch.Tensor):
            text_embs = text_embs.pooler_output
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = self.logit_scale.exp() * torch.diagonal(text_embs @ image_embs.T)
        return {'reward_pred': scores}
