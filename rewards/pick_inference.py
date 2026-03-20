"""
PickScore implementation for image-text alignment scoring.
Based on: https://github.com/yuvalkirstain/PickScore
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Union


class PickScoreModel:
    """
    PickScore model wrapper that provides ImageReward-like interface.
    """

    def __init__(self, device="cuda"):
        """
        Initialize PickScore model.

        Args:
            device: Device to run the model on (default: "cuda")
        """
        self.device = device
        print(f"Loading PickScore model on {device}...")

        # Load processor and model
        self.processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.model = CLIPModel.from_pretrained(
            "yuvalkirstain/PickScore_v1"
        ).eval().to(device)

        print("PickScore model loaded successfully!")

    def __call__(self, prompt: str, images: List[Image.Image]) -> List[float]:
        """
        Score images based on their alignment with the text prompt.

        Args:
            prompt: Text prompt (single string)
            images: List of PIL Image objects

        Returns:
            List of scores (floats) for each image
        """
        # Prepare inputs — process text+images together, then split
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Calculate scores
        with torch.no_grad():
            # Get image embeddings (handle both tensor and BaseModelOutput returns)
            image_embs = self.model.get_image_features(**image_inputs)
            if not isinstance(image_embs, torch.Tensor):
                image_embs = image_embs.pooler_output
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            # Get text embeddings
            text_embs = self.model.get_text_features(**text_inputs)
            if not isinstance(text_embs, torch.Tensor):
                text_embs = text_embs.pooler_output
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # Calculate similarity scores
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        # Return scores as list of floats
        return scores.cpu().tolist()

