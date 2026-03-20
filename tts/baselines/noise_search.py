import torch
import numpy as np

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info

@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           n=6, seed=42, m=4, k=0, r=0.15, epsilon=0.0,
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, use_repel=False):

    """Noise-space search. BON = k=0, greedy = k>0.

    1. Generate n candidates from random noise, pick the best.
    2. For k iterations: sample m neighbours around the best (radius r),
       update best if any neighbour improves the score.

    Args:
        pipe:                loaded Flux pipeline
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        n:                   number of initial candidates
        seed:                random seed
        m:                   neighbours per greedy iteration
        k:                   greedy iterations (0 = BON)
        r:                   perturbation radius (0=identical, 1=fully fresh)
        epsilon:             per-candidate probability of sampling fresh noise instead of perturbing best (0=greedy)
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: denoising steps

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """

    torch.manual_seed(seed)
    if res == 512:
        noise = torch.randn((n, 1024, 64), dtype=torch.bfloat16)
    elif res == 1024:
        noise = torch.randn((n, 4096, 64), dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported resolution: {res}")
    _, images = gen_images(pipe, prompt, noise, start_step=0, end_step=None,
                           res=res, num_inference_steps=num_inference_steps,
                           batch_size=gen_batch_size, use_repel=use_repel)
    scores, ensemble_raw = score_images(reward_name, reward_model, prompt, images, reward_batch_size)

    best_idx   = int(np.argmax(scores))
    best_noise = noise[best_idx]
    best_image = images[best_idx]
    best_score = float(scores[best_idx])
    ensemble_score_info = extract_ensemble_info(ensemble_raw, best_idx)

    for _ in range(k):
        fresh    = torch.randn((m, *best_noise.shape), dtype=torch.bfloat16)
        is_rand  = torch.tensor(np.random.rand(m) < epsilon).float().view(m, 1, 1)
        base     = is_rand * torch.randn((m, *best_noise.shape), dtype=torch.bfloat16) \
                 + (1 - is_rand) * best_noise.unsqueeze(0)
        new_noise = (1.0 - r ** 2) ** 0.5 * base + r * fresh
        _, new_imgs = gen_images(pipe, prompt, new_noise, start_step=0, end_step=None,
                                 res=res, num_inference_steps=num_inference_steps,
                                 batch_size=gen_batch_size, use_repel=use_repel)
        new_scores, new_ensemble_raw = score_images(reward_name, reward_model, prompt, new_imgs, reward_batch_size)

        local_best = int(np.argmax(new_scores))
        if new_scores[local_best] > best_score:
            best_noise = new_noise[local_best]
            best_image = new_imgs[local_best]
            best_score = float(new_scores[local_best])
            ensemble_score_info = extract_ensemble_info(new_ensemble_raw, local_best)

    return best_image, best_score, ensemble_score_info
