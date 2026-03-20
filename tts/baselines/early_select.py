import torch
import numpy as np

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info

@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           seed=42, n=6, d=10,
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, use_repel=False):
    """Early-selection search (ODE).

    Denoise N candidates in parallel to step d, score their x0 predictions,
    pick the best, then finish denoising that single candidate to completion.

    Args:
        pipe:                loaded Flux pipeline (ODE scheduler)
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        seed:                random seed
        n:                   number of initial candidates
        d:                   early-selection step (must be < num_inference_steps)
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: total denoising steps

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """
    assert d < num_inference_steps, f"d ({d}) must be < num_inference_steps ({num_inference_steps})"

    torch.manual_seed(seed)
    candidates = torch.randn(n, 1024, 64, dtype=torch.bfloat16)

    # Denoise all N from 0 → d, score x0 predictions, pick the best
    cand_lats, cand_imgs = gen_images(pipe, prompt, candidates,
                                      start_step=0, end_step=d,
                                      res=res, num_inference_steps=num_inference_steps,
                                      batch_size=gen_batch_size, use_repel=use_repel)
    scores, _ = score_images(reward_name, reward_model, prompt, cand_imgs, reward_batch_size, step=d)
    best_idx  = int(np.argmax(scores))

    # Keep only the best latent, free the rest
    best_lat = cand_lats[best_idx:best_idx + 1].clone()
    del candidates, cand_lats, cand_imgs
    torch.cuda.empty_cache()

    # Finish denoising the best latent from d → num_inference_steps
    _, final_imgs = gen_images(pipe, prompt, best_lat,
                               start_step=d, end_step=num_inference_steps,
                               res=res, num_inference_steps=num_inference_steps,
                               batch_size=1, use_repel=use_repel)
    final_scores, ensemble_raw = score_images(reward_name, reward_model, prompt, final_imgs, reward_batch_size)
    ensemble_score_info = extract_ensemble_info(ensemble_raw, 0)

    return final_imgs[0], float(final_scores[0]), ensemble_score_info
