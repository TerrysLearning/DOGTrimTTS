"""
Thin wrapper around hpsv2.inference to provide a consistent .score(prompt, images) interface.
"""

import os
import sys
from PIL import Image
from typing import List

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from hpsv2.inference import setup_model, score


class HPSv2Model:
    def __init__(self, device="cuda", version="v2.1"):
        print(f"Loading HPSv2 model (version: {version}) on {device}...")
        self.model_dict = setup_model(hps_version=version, device=device)
        print("HPSv2 model loaded successfully!")

    def __call__(self, prompt: str, images: List[Image.Image]) -> List[float]:
        return score(self.model_dict, images, prompt)
