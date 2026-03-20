import ImageReward as RM


class ImageReward_Model:
    def __init__(self, device='cuda'):
        print(f"Loading ImageReward model on {device}...")
        self.model = RM.load("ImageReward-v1.0", device=device)
        print("ImageReward model loaded successfully!")

    def __call__(self, prompt: str, images) -> list:
        scores = self.model.score_image_list(prompt, images)
        return scores
