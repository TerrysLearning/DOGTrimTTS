import os
import json
import random
import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    """Pair dataset for Bradley-Terry preference loss.

    Each __getitem__ returns a pair of images from the same prompt,
    ordered so the first has higher GT score (target=0 for cross-entropy).

    The `current_step` attribute must be set by the training loop before
    each curriculum phase (e.g. dataset.current_step = 28).
    """

    def __init__(self, gt_file, prompts_file, data_folder, prompt_range, num_per_prompt):
        """
        Args:
            gt_file: path to label JSON, e.g. labels/xxx.json
            prompts_file: path to prompts JSON list
            data_folder: root image folder
            prompt_range: use first N prompts
            num_per_prompt: number of images per prompt
        """
        self.data_folder = data_folder
        self.num_per_prompt = num_per_prompt
        self.current_step = None  # set by training loop (image timestep)
        self.gt_step = 35         # GT reward timestep (base_model_time); override if needed

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

    def _get_score(self, p_id, img_id):
        gt_key = f"d{self.gt_step}"
        score = self.gt_dict[f"p{p_id:05d}"][f"img{img_id:03d}"].get(gt_key, 0.0)
        return score

    def __getitem__(self, idx):
        assert self.current_step is not None, "Set dataset.current_step before training"
        p_id, img_id = self.data_tuples[idx]
        prompt = self.prompts_info[p_id]
        score = self._get_score(p_id, img_id)

        # Pick a random different image from the same prompt
        other_ids = [x for x in range(self.num_per_prompt) if x != img_id]
        img_id_other = random.choice(other_ids)
        score_other = self._get_score(p_id, img_id_other)

        # Order: first has higher score (Bradley-Terry target=0)
        if score >= score_other:
            ids = [img_id, img_id_other]
            scores = [score, score_other]
        else:
            ids = [img_id_other, img_id]
            scores = [score_other, score]

        return {
            'prompt': [prompt, prompt],
            'reward_gt': torch.tensor(scores, dtype=torch.float32),
            'p_id': torch.tensor([p_id, p_id], dtype=torch.long),
            'img_id': torch.tensor(ids, dtype=torch.long),
        }


def collate_fn(batch):
    prompts = []
    for item in batch:
        prompts.extend(item['prompt'])
    return {
        'prompt': prompts,
        'reward_gt': torch.cat([item['reward_gt'] for item in batch]),
        'p_id': torch.cat([item['p_id'] for item in batch]),
        'img_id': torch.cat([item['img_id'] for item in batch]),
    }
