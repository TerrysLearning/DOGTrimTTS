import torch
import numpy as np

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info

@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           seed=42, n=4, d=7,
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, use_repel=False):
    """Trajectory greedy search (local-path).

    Start from one noise replicated n times. At each stage the SDE scheduler
    injects different noise via SDE into each copy, producing n diverse candidates.
    Pick the best, replicate again, repeat until fully denoised.

    Args:
        pipe:                loaded Flux pipeline (SDE scheduler)
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        seed:                random seed
        n:                   candidates per stage
        d:                   denoising steps per stage
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: total denoising steps

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """
    torch.manual_seed(seed)
    candidates = torch.randn(1, 1024, 64, dtype=torch.bfloat16).repeat(n, 1, 1)

    current_step        = 0
    best_score          = -float('inf')
    best_image          = None
    ensemble_score_info = None

    while current_step < num_inference_steps:
        next_step = min(current_step + d, num_inference_steps)

        cand_lats, cand_imgs = gen_images(pipe, prompt, candidates,
                                          start_step=current_step, end_step=next_step,
                                          res=res, num_inference_steps=num_inference_steps,
                                          batch_size=gen_batch_size, use_repel=use_repel)
        scores, ensemble_raw = score_images(reward_name, reward_model, prompt, cand_imgs, reward_batch_size, step=next_step)
        best_idx = int(np.argmax(scores))
        best_score          = float(scores[best_idx])
        best_image          = cand_imgs[best_idx]
        ensemble_score_info = extract_ensemble_info(ensemble_raw, best_idx)
        current_step        = next_step

        # replicate best latent n times for the next stage
        candidates = cand_lats[best_idx:best_idx + 1].repeat(n, 1, 1)

    return best_image, best_score, ensemble_score_info
