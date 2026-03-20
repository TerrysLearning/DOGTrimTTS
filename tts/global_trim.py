import torch
import numpy as np

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info


@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           seed=42, n=16, d=7, gamma=0.6,
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, use_repel=False):
    """Global trimming search.

    Start with N candidates. At each stage denoise d steps, score the x0
    predictions, and prune the bottom candidates keeping the top
    ceil(n_alive * gamma) survivors. Repeat until fully denoised, then
    return the best surviving image.

    Args:
        pipe:                loaded Flux pipeline (ODE scheduler)
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        seed:                random seed
        n:                   number of initial candidates
        d:                   denoising steps per stage
        gamma:               fraction of candidates to keep at each stage (0 < gamma <= 1)
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: total denoising steps

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """
    assert 0.0 < gamma <= 1.0, f"gamma ({gamma}) must be in (0, 1]"

    torch.manual_seed(seed)
    candidates = torch.randn(n, 1024, 64, dtype=torch.bfloat16)

    current_step        = 0
    best_image          = None
    best_score          = -float('inf')
    ensemble_score_info = None

    while current_step < num_inference_steps:
        next_step = min(current_step + d, num_inference_steps)
        n_alive   = candidates.shape[0]

        cand_lats, cand_imgs = gen_images(pipe, prompt, candidates,
                                          start_step=current_step, end_step=next_step,
                                          res=res, num_inference_steps=num_inference_steps,
                                          batch_size=gen_batch_size, use_repel=use_repel)
        scores, ensemble_raw = score_images(reward_name, reward_model, prompt, cand_imgs, reward_batch_size, step=next_step)
        scores_np = np.array(scores, dtype=np.float64)

        best_idx            = int(np.argmax(scores_np))
        best_score          = float(scores_np[best_idx])
        best_image          = cand_imgs[best_idx]
        ensemble_score_info = extract_ensemble_info(ensemble_raw, best_idx)
        current_step        = next_step

        if current_step >= num_inference_steps:
            break

        # Prune: keep top ceil(n_alive * gamma) candidates
        keep_n     = max(1, round(n_alive * gamma))
        top_idx    = np.argsort(-scores_np)[:keep_n]
        candidates = cand_lats[torch.tensor(top_idx, dtype=torch.long)]

    return best_image, best_score, ensemble_score_info
