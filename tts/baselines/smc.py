import torch
import numpy as np
from numpy import random as np_random

from utils_flux import gen_images
from utils_reward import score_images, extract_ensemble_info


# ============================================================
# SSP / multinomial resampling helpers
# ============================================================
def ssp(W, M):
    """SSP resampling (Srinivasan Sampling Process). Returns M indices (may contain duplicates)."""
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    u = np_random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])
        delta_j = min(xi[i], 1.0 - xi[j])
        sum_delta = delta_i + delta_j
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        if u[k] < pj:
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        raise ValueError("ssp resampling: wrong size for output")
    return np.arange(N).repeat(nr_children)


def multinomial_resample(W, M):
    """Multinomial resampling. Draw M indices from distribution W."""
    return np_random.choice(len(W), size=M, p=W)


def _log_normalize(log_w):
    """Log-sum-exp normalization: returns log-probabilities."""
    log_w = log_w - log_w.max()
    log_w = log_w - np.log(np.sum(np.exp(log_w)))
    return log_w


def compute_weights(scores, population_rs, lmbda, tempering, potential, stage, num_stages):
    """Compute normalized importance weights.

    Args:
        scores:       current-stage scores, shape (N,)
        population_rs: accumulated reward per particle, shape (N,)
        lmbda:        softmax temperature
        tempering:    'increase' (scores scaled by stage/num_stages*2) or 'constant'
        potential:    'max' | 'add' | 'diff' | 'rt'
        stage:        current stage index (0-based)
        num_stages:   total number of stages

    Returns:
        (w, effective_r): normalized weights and updated accumulated rewards
    """
    scores = np.array(scores, dtype=np.float64)

    if tempering == 'increase':
        scores = scores * (stage / num_stages * 2)
    elif tempering != 'constant':
        raise ValueError(f"Unknown tempering schedule: {tempering}")

    if potential == 'max':
        effective_r = np.maximum(scores, population_rs)
    elif potential == 'add':
        effective_r = scores + population_rs
    elif potential == 'diff':
        effective_r = scores - population_rs
    elif potential == 'rt':
        effective_r = scores
    else:
        raise ValueError(f"Unknown potential type: {potential}")

    log_w = lmbda * effective_r
    w = np.exp(_log_normalize(log_w))
    return w, effective_r


# ============================================================
# SMC-SDE search
# ============================================================
@torch.no_grad()
def search(pipe, reward_model, reward_name, prompt,
           seed=42, n=6, d=7, lmbda=50.0,
           tempering='increase', potential='max', resample='ssp',
           gen_batch_size=8, reward_batch_size=16,
           res=512, num_inference_steps=35, ode_case=False, use_repel=False):
    """SMC-SDE: Sequential Monte Carlo with SDE scheduler and full resampling.

    N particles are maintained throughout. At each stage the particles are
    denoised d steps, scored, and resampled (SSP or multinomial) so the
    population stays at N. Because the SDE scheduler injects noise, duplicated
    candidates naturally diverge in the next stage.

    Args:
        pipe:                loaded Flux pipeline (SDE scheduler)
        reward_model:        loaded reward model
        reward_name:         'pickscore' | 'imagereward' | 'hpsv2' | 'ensemble'
        prompt:              text prompt
        seed:                random seed
        n:                   number of particles
        d:                   denoising steps per stage
        lmbda:               softmax temperature for importance weights
        tempering:           'increase' (scores scaled up over stages) or 'constant'
        potential:           'max' | 'add' | 'diff' | 'rt'
        resample:            'ssp' | 'multinomial'
        gen_batch_size:      max images per generation call
        reward_batch_size:   max images per reward scoring call
        res:                 image resolution
        num_inference_steps: total denoising steps
        ode_case:            if True, deduplicate resampled indices (pruning only);
                             use with an ODE scheduler since there is no SDE noise
                             to diverge duplicated particles

    Returns:
        (best_image: PIL.Image, best_score: float, ensemble_score_info: list[float] | None)
        ensemble_score_info is [hps, pick, imr] for the best image when reward_name=='ensemble', else None.
    """
    torch.manual_seed(seed)
    candidates = torch.randn(n, 1024, 64, dtype=torch.bfloat16)

    num_stages    = -(-num_inference_steps // d)
    population_rs = np.zeros(n, dtype=np.float64)

    current_step        = 0
    stage               = 0
    best_image          = None
    best_score          = -float('inf')
    ensemble_score_info = None

    while current_step < num_inference_steps:
        next_step = min(current_step + d, num_inference_steps)

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

        # Resample: keep N candidates (with duplication); SDE diverges duplicates
        w, effective_r = compute_weights(scores_np, population_rs,
                                         lmbda, tempering, potential,
                                         stage, num_stages)
        if resample == 'ssp':
            resample_idx = ssp(w, n)
        else:
            resample_idx = multinomial_resample(w, n)

        if ode_case:
            # ODE: no SDE noise to diverge duplicates, so keep only unique particles (pruning)
            resample_idx = np.unique(resample_idx)
        candidates    = cand_lats[torch.tensor(resample_idx, dtype=torch.long)]
        population_rs = effective_r[resample_idx]
        stage += 1

    return best_image, best_score, ensemble_score_info
