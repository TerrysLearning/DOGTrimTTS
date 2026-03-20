import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from rewards.pick_inference import PickScoreModel
from rewards.hps_inference import HPSv2Model
from rewards.imr_inference import ImageReward_Model


def load_reward(reward_name, device='cuda'):
    print(f"Loading reward model: {reward_name}")
    if reward_name == 'pickscore':
        return PickScoreModel(device=device)
    elif reward_name == 'imagereward':
        return ImageReward_Model(device=device)
    elif reward_name == 'hpsv2':
        return HPSv2Model(device=device, version='v2.1')
    elif reward_name == 'ensemble':
        return {
            'pickscore':   PickScoreModel(device=device),
            'imagereward': ImageReward_Model(device=device),
            'hpsv2':       HPSv2Model(device=device, version='v2.1'),
        }
    else:
        raise ValueError(f"Unknown reward: {reward_name}")


# ── NARF ──────────────────────────────────────────────────────────────────────

class NARFReward:
    """Wraps a reward model with step-specific NARF checkpoints.

    State dicts are pre-loaded to CPU. At scoring time, set_step() swaps in
    the appropriate weights (or falls back to the base model's original weights).
    """

    def __init__(self, base_model, reward_name, ckpt_dir):
        self.model = base_model
        self._inner = self._get_inner(base_model, reward_name)
        self._base_state = {k: v.cpu().clone() for k, v in self._inner.state_dict().items()}
        self._current_step = None

        # Pre-load all step{NN}.pt to CPU; filename pattern: step{step}.pt
        sub = os.path.join(ckpt_dir, reward_name)
        self.step_states = {}
        if os.path.isdir(sub):
            for fname in sorted(os.listdir(sub)):
                if fname.endswith('.pt') and fname.startswith('step'):
                    step = int(fname[4:-3])
                    self.step_states[step] = torch.load(os.path.join(sub, fname), map_location='cpu', weights_only=False)
                    print(f"[NARF] Loaded checkpoint for step {step} from {fname}")
        print(f"[NARF] {reward_name}: checkpoints for steps {sorted(self.step_states.keys())}")

    @staticmethod
    def _get_inner(model, reward_name):
        if reward_name == 'hpsv2':
            return model.model_dict['model']
        elif reward_name in ('pickscore', 'imagereward'):
            return model.model
        raise ValueError(f"NARF not supported for: {reward_name}")

    def set_step(self, step):
        if step == self._current_step:
            return
        self._current_step = step
        if step in self.step_states:
            self._inner.load_state_dict(self.step_states[step], strict=False)
            print(f"[NARF] Use checkpoint for step {step}")
        else:
            print(f"[NARF] No checkpoint for step {step}, using base model")
            self._inner.load_state_dict(self._base_state, strict=False)

    def __call__(self, prompt, images):
        return self.model(prompt, images)


def wrap_narf(reward_model, reward_name, ckpt_dir):
    """Wrap a loaded reward model (or ensemble dict) with NARF checkpoints."""
    if reward_name == 'ensemble':
        return {k: NARFReward(v, k, ckpt_dir) for k, v in reward_model.items()}
    return NARFReward(reward_model, reward_name, ckpt_dir)


# ── internal helpers ──────────────────────────────────────────────────────────

def score_one_reward(model, prompt, images, batch_size):
    all_scores = []
    for i in range(0, len(images), batch_size):
        all_scores.extend(model(prompt, images[i:i + batch_size]))
    return all_scores


def ensemble_rank_sum(scores_list):
    n = len(scores_list[0])
    rank_sums = np.zeros(n, dtype=np.float64)
    for scores in scores_list:
        arr = np.array(scores, dtype=np.float64)
        rank_sums += np.argsort(np.argsort(arr)) + 1
    return rank_sums.tolist()


def extract_ensemble_info(ensemble_raw, idx):
    if ensemble_raw is None:
        return None
    return [ensemble_raw['hps'][idx], ensemble_raw['pick'][idx], ensemble_raw['imr'][idx]]


# ── public API ────────────────────────────────────────────────────────────────

def score_images(reward_name, reward_model, prompt, images, batch_size=16, step=None):
    """Score PIL images. Pass step= for NARF step-aware scoring."""
    if step is not None:
        models = reward_model.values() if reward_name == 'ensemble' else [reward_model]
        for m in models:
            if hasattr(m, 'set_step'):
                m.set_step(step)

    if reward_name == 'ensemble':
        hps  = score_one_reward(reward_model['hpsv2'],       prompt, images, batch_size)
        pick = score_one_reward(reward_model['pickscore'],   prompt, images, batch_size)
        ir   = score_one_reward(reward_model['imagereward'], prompt, images, batch_size)
        return ensemble_rank_sum([hps, pick, ir]), {'hps': hps, 'pick': pick, 'imr': ir}

    return score_one_reward(reward_model, prompt, images, batch_size), None
