import torch
import numpy as np

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info

@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           seed=42, m=4, d_b=7, d_f=3,
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, use_repel=False):
    """Search over Path (SoP).

    1. Denoise a single noise from step 0 to d_b.
    2. Re-noise the best x0 back d_f steps → m candidates.
    3. Denoise each candidate d_b steps further, score via x0 prediction.
    4. Keep the best, repeat until num_inference_steps is reached.
    Net progress per iteration: d_b - d_f steps.

    Args:
        pipe:                loaded Flux pipeline
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        seed:                random seed
        m:                   candidate continuations per iteration
        d_b:                 denoising steps per leg (delta backward)
        d_f:                 re-noising steps per iteration (delta forward), must be < d_b
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: total denoising steps

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """
    assert d_f < d_b, f"d_f ({d_f}) must be < d_b ({d_b})"

    torch.manual_seed(seed)
    init_noise = torch.randn(1, 1024, 64, dtype=torch.bfloat16)

    # Phase 0: initial denoise 0 → d_b, get x0 as latent for re-noising
    best_lat, best_x0 = gen_images(pipe, prompt, init_noise,
                                   start_step=0, end_step=d_b,
                                   res=res, num_inference_steps=num_inference_steps,
                                   batch_size=1, output_type="latent", use_repel=use_repel)
    # best_imgs = pipe.decode_latents(best_x0, res, res)
    # scores, ensemble_raw = score_images(reward_name, reward_model, prompt, best_imgs, reward_batch_size)
    # best_score = float(scores[0])
    # ensemble_score_info = extract_ensemble_info(ensemble_raw, 0)
    current_step = d_b

    while current_step < num_inference_steps:
        target_step = current_step - d_f
        next_step   = min(current_step + (d_b - d_f), num_inference_steps)

        # Re-noise x0 latent back to target_step with m different noise vectors
        candidates = torch.cat([
            pipe.scheduler.add_noise_at_step(best_x0, target_step,
                                             torch.randn_like(best_x0))
            for _ in range(m)
        ], dim=0)

        cand_lats, cand_x0_latents = gen_images(pipe, prompt, candidates,
                                               start_step=target_step, end_step=next_step,
                                               res=res, num_inference_steps=num_inference_steps,
                                               batch_size=gen_batch_size,
                                               output_type="latent", use_repel=use_repel)
        cand_imgs = pipe.decode_latents(cand_x0_latents, res, res)
        scores, ensemble_raw = score_images(reward_name, reward_model, prompt, cand_imgs, reward_batch_size, step=next_step)
        best_idx = int(np.argmax(scores))

        best_lat   = cand_lats[best_idx:best_idx + 1]
        best_x0    = cand_x0_latents[best_idx:best_idx + 1]
        best_score = float(scores[best_idx])
        ensemble_score_info = extract_ensemble_info(ensemble_raw, best_idx)
        current_step = next_step

    return cand_imgs[best_idx], best_score, ensemble_score_info
