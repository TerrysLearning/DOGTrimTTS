import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kendalltau as scipy_kendalltau

from utils import load_image_batch


class EvalDataset(Dataset):
    """Flat dataset for evaluation — returns individual images (no pairing)."""

    def __init__(self, gt_file, prompts_file, data_folder, prompt_range, num_per_prompt):
        self.data_folder = data_folder
        self.num_per_prompt = num_per_prompt
        self.current_step = None  # timestep for image loading; set before evaluate()
        self.gt_step = None       # timestep for GT lookup (base_model_time); set before evaluate()

        with open(gt_file, 'r') as f:
            self.gt_dict = json.load(f)
        with open(prompts_file, 'r') as f:
            prompts_list = json.load(f)

        self.prompts_info = {}
        self.data_tuples = []  # [(p_id, img_id)]
        for p_id in range(min(prompt_range, len(prompts_list))):
            p_key = f"p{p_id:05d}"
            if p_key not in self.gt_dict:
                continue
            self.prompts_info[p_id] = prompts_list[p_id]
            for img_id in range(num_per_prompt):
                self.data_tuples.append((p_id, img_id))

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        assert self.current_step is not None, "Set dataset.current_step before evaluating"
        assert self.gt_step is not None, "Set dataset.gt_step before evaluating"
        p_id, img_id = self.data_tuples[idx]
        gt_key = f"d{self.gt_step}"
        gt = self.gt_dict[f"p{p_id:05d}"][f"img{img_id:03d}"].get(gt_key, 0.0)
        return {
            'prompt': self.prompts_info[p_id],
            'reward_gt': torch.tensor(gt, dtype=torch.float32),
            'p_id': torch.tensor(p_id, dtype=torch.long),
            'img_id': torch.tensor(img_id, dtype=torch.long),
        }


def _eval_collate_fn(batch):
    return {
        'prompt': [item['prompt'] for item in batch],
        'reward_gt': torch.stack([item['reward_gt'] for item in batch]),
        'p_id': torch.stack([item['p_id'] for item in batch]),
        'img_id': torch.stack([item['img_id'] for item in batch]),
    }


def evaluate(model, accelerator, args_data, step, gt_step, batch_size=16):
    """Compute mean Kendall's tau between GT and predicted reward scores.

    For each prompt, computes Kendall's tau across all images.
    Images are loaded at `step` (the current curriculum timestep).
    GT rewards are always looked up at `gt_step` (base_model_time, e.g. 35),
    i.e. the final fully-denoised reward.

    Sanity check: when step == gt_step, tau should be ~1.0 (model predicts
    the same reward it was trained on, from the same images).

    Args:
        model: reward model (already prepared by accelerator)
        accelerator: Accelerator instance
        args_data: data config dict from yaml
        step: timestep for image loading (e.g. 7, 14, 21, 28, 35)
        gt_step: timestep for GT reward lookup (base_model_time from config)
        batch_size: inference batch size per GPU

    Returns:
        mean Kendall's tau (float) on main process, None on other processes
    """
    gt_file      = args_data.get('valid_gt_file')      or args_data['train_gt_file']
    prompts_file = args_data.get('valid_prompts_file') or args_data['train_prompts_file']
    data_folder  = args_data.get('valid_data_folder')  or args_data['train_data_folder']
    prompt_range = args_data.get('valid_prompt_range') or args_data['train_prompt_range']
    num_per_prompt = args_data.get('valid_num_per_prompt') or args_data['train_num_per_prompt']

    eval_dataset = EvalDataset(
        gt_file=gt_file,
        prompts_file=prompts_file,
        data_folder=data_folder,
        prompt_range=prompt_range,
        num_per_prompt=num_per_prompt,
    )
    eval_dataset.current_step = step
    eval_dataset.gt_step = gt_step

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_eval_collate_fn,
        num_workers=0,
    )
    eval_loader = accelerator.prepare(eval_loader)

    model.eval()
    all_p_ids, all_preds, all_gts = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            batch['image'] = load_image_batch(
                data_folder,
                batch['p_id'], batch['img_id'],
                t_id=step,
            ).to(accelerator.device)

            pred = model(batch)
            all_p_ids.append(batch['p_id'])
            all_preds.append(pred['reward_pred'].float())
            all_gts.append(batch['reward_gt'].to(accelerator.device))

    # Gather results from all GPUs
    all_p_ids = accelerator.gather_for_metrics(torch.cat(all_p_ids))
    all_preds = accelerator.gather_for_metrics(torch.cat(all_preds))
    all_gts = accelerator.gather_for_metrics(torch.cat(all_gts))

    model.train()

    if not accelerator.is_main_process:
        return None

    p_ids = all_p_ids.cpu().numpy()
    preds = all_preds.cpu().numpy()
    gts = all_gts.cpu().numpy()

    # Compute Kendall's tau per prompt
    taus = []
    for pid in np.unique(p_ids):
        mask = p_ids == pid
        if mask.sum() < 2:
            continue
        # print(gts[mask])
        # print(preds[mask])
        # print("///////")
        tau, _ = scipy_kendalltau(gts[mask], preds[mask])
        if not np.isnan(tau):
            taus.append(tau)

    n_prompts = len(taus)
    n_total = int(len(np.unique(p_ids)))
    mean_tau = float(np.mean(taus)) if taus else float('nan')
    print(f"  Eval d{step} (gt=d{gt_step}): Kendall tau = {mean_tau:.4f}  "
          f"({n_prompts}/{n_total} prompts)")
    return {"tau": mean_tau, "n_prompts": n_prompts, "n_total": n_total}
