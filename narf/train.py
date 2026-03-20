"""
Curriculum reward training with Bradley-Terry preference loss.
Multi-GPU via Accelerate.

Usage:
    accelerate launch train.py --config configs/config_hps.yaml
    accelerate launch --num_processes=2 train.py --config configs/config_hps.yaml
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from dataset import RewardDataset, collate_fn
from utils import load_image_batch, get_scheduler
from eval import evaluate


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(reward_name, device='cpu'):
    if reward_name == 'hps':
        from hps import HPS_Model
        model = HPS_Model(device=device)
        return model

    elif reward_name == 'pick':
        from pick import PickScore_Model
        model = PickScore_Model.from_pretrained("yuvalkirstain/PickScore_v1")
        model.setup_processor()
        return model

    elif reward_name == 'imr':
        from imr import ImageReward_Model
        model = ImageReward_Model()
        return model

    else:
        raise ValueError(f"Unknown reward model: {reward_name}")


def load_base_weights(model, reward_name):
    """Load pretrained base weights from online sources."""
    import huggingface_hub

    if reward_name == 'hps':
        from hpsv2.utils import hps_version_map
        ckpt_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  HPS load — missing: {missing}, unexpected: {unexpected}")
        print("  HPS base weights loaded from xswu/HPSv2 (v2.1)")

    elif reward_name == 'pick':
        # Weights already loaded via from_pretrained("yuvalkirstain/PickScore_v1") in setup_model
        print("  PickScore base weights loaded from yuvalkirstain/PickScore_v1")

    elif reward_name == 'imr':
        cache_dir = os.path.expanduser("~/.cache/ImageReward")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = huggingface_hub.hf_hub_download(
            repo_id="THUDM/ImageReward", filename="ImageReward.pt", local_dir=cache_dir
        )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  IMR load — missing: {missing}, unexpected: {unexpected}")
        print("  ImageReward base weights loaded from THUDM/ImageReward")


def preference_loss(pred, tau=10):
    """Bradley-Terry preference loss.
    pred: [B] where B is even. Pairs are consecutive: (winner, loser).
    """
    pred_pairs = pred.reshape(-1, 2)
    target = torch.zeros(pred_pairs.shape[0], dtype=torch.long, device=pred.device)
    return F.cross_entropy(tau * pred_pairs, target, reduction='mean')


def mse_loss(pred, gt):
    """MSE loss against ground truth reward scores."""
    return F.mse_loss(pred, gt.to(pred.device), reduction='mean')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    config = parse_config(args.config)
    reward_name = config['reward_name']
    seed = config['seed']
    args_data = config['data']
    args_train = config['train']
    args_log = config['log']
    step_targets = sorted(config['time_config']['step_targets'], reverse=True)

    # Seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Accelerator (create before model so device placement is correct)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args_train.get('accum_steps', 1),
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    # Dataset (batch_size // 2 because each item returns a pair)
    batch_size_train = args_train['batch_size'] // 2
    dataset = RewardDataset(
        gt_file=args_data['train_gt_file'],
        prompts_file=args_data['train_prompts_file'],
        data_folder=args_data['train_data_folder'],
        prompt_range=args_data['train_prompt_range'],
        num_per_prompt=args_data['train_num_per_prompt'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Model — load on CPU, then accelerator.prepare() moves to correct GPU
    if accelerator.is_main_process:
        print(f"Setting up {reward_name} model...")
    model = setup_model(reward_name, device='cpu')
    model.to("cuda")
    load_base_weights(model, reward_name)
    model.requires_grad_(True)
  

    model, dataloader = accelerator.prepare(model, dataloader)

    # Checkpoint dir
    os.makedirs(args_log.get('checkpoint_dir'), exist_ok=True)
    ckpt_dir = os.path.join(
        args_log.get('checkpoint_dir', 'out_ckpt'),
        args_log['name']
    )

    gt_step = args_train.get('base_model_time', 35)
    dataset.gt_step = gt_step  # GT reward always from base_model_time

    # Output logs
    exp_name = args_log['name']
    out_evals_dir = 'out_evals'
    eval_json_path = os.path.join(out_evals_dir, f'{exp_name}_eval.json')
    loss_txt_path  = os.path.join(out_evals_dir, f'{exp_name}_loss.txt')
    eval_log = []
    if accelerator.is_main_process:
        os.makedirs(out_evals_dir, exist_ok=True)

    def _save_eval(entry):
        eval_log.append(entry)
        with open(eval_json_path, 'w') as f:
            json.dump(eval_log, f, indent=2)

    def run_eval(step, phase, curr_step=None, epoch=None):
        if args_log.get('do_valid', False):
            if accelerator.is_main_process:
                print(f"  Running eval for d{step} (gt=d{gt_step})...")
            result = evaluate(model, accelerator, args_data, step, gt_step=gt_step,
                              batch_size=args_train.get('batch_size', 16))
            if accelerator.is_main_process and result is not None:
                _save_eval({
                    "phase": phase,
                    "curr_step": curr_step,
                    "epoch": epoch,
                    "eval_step": step,
                    "gt_step": gt_step,
                    **result,
                })

    # Evaluate before training if requested
    if args_log.get('eval_start', False):
        if accelerator.is_main_process:
            print("\n--- Pre-training evaluation ---")
        for step in step_targets:
            run_eval(step, phase='eval_start')
        accelerator.wait_for_everyone()

    # Curriculum training
    epochs_total = 0
    for i, step in enumerate(step_targets):
        # Set dataset timestep
        dataset.current_step = step

        # Decaying initial LR across curriculum steps
        lr_decay = 1.0 - (1.0 - args_train.get('min_lr_scale', 0.5)) * i / max(1, len(step_targets))
        lr_start = float(args_train['lr']) * lr_decay

        if accelerator.is_main_process:
            print(f"\n{'#' * 60}")
            print(f"Curriculum step {i+1}/{len(step_targets)}: d{step}, lr={lr_start:.2e}")
            print(f"{'#' * 60}")

        # New optimizer + scheduler per curriculum step
        if i > 0:
            del optimizer, scheduler
            accelerator.free_memory()

        total_steps = len(dataloader) * args_train['curr_interval']
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_start,
            betas=(float(args_train.get('adam_beta1', 0.9)),
                   float(args_train.get('adam_beta2', 0.999))),
            eps=float(args_train.get('adam_eps', 1e-8)),
            weight_decay=float(args_train.get('weight_decay', 0.01)),
        )
        scheduler = get_scheduler(
            optimizer, total_steps,
            warmup_ratio=float(args_train.get('warmup_ratio', 0)),
            warmup_type=args_train.get('warmup_type', 'linear'),
            scheduler_type=args_train.get('scheduler_type', 'cosine'),
            min_lr_scale=float(args_train.get('min_lr_scale', 0.0)),
        )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        model.train()
        for epoch in range(args_train['curr_interval']+ i * args_train['curr_interval_add'] ):
            for step_i, batch in enumerate(dataloader):
                # Load images at current curriculum timestep
                batch['image'] = load_image_batch(
                    args_data['train_data_folder'],
                    batch['p_id'], batch['img_id'],
                    t_id=step,
                ).to(accelerator.device)

                with accelerator.accumulate(model):
                    pred = model(batch)
                    reward_pred = pred['reward_pred']

                    loss = torch.tensor(0.0, device=accelerator.device)
                    loss_info = {}
                    if args_train.get('use_BT_loss', True):
                        l_bt = preference_loss(reward_pred, tau=float(args_train.get('loss_preference_tau', 10)))
                        loss = loss + l_bt
                        loss_info['BT'] = l_bt.item()
                    if args_train.get('use_MSE_loss', False):
                        l_mse = mse_loss(reward_pred, batch['reward_gt'].float())
                        loss = loss + l_mse
                        loss_info['MSE'] = l_mse.item()

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(args_train.get('max_grad_norm', 5.0)),
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                if accelerator.is_main_process:
                    if (step_i + 1) % args_log.get('print_loss_every_iter', 1) == 0:
                        loss_str = ' | '.join(f'{k}={v:.6f}' for k, v in loss_info.items())
                        loss_line = (f"  epoch {epochs_total}, step {step_i+1}: "
                                     f"loss={loss.item():.6f} [{loss_str}], lr={scheduler.get_last_lr()[0]:.2e}")
                        print(loss_line)
                        with open(loss_txt_path, 'a') as f:
                            f.write(loss_line + '\n')

            epochs_total += 1

            # Evaluate at current and all subsequent curriculum steps
            for eval_step in step_targets[i:]:
                run_eval(eval_step, phase='curriculum', curr_step=step, epoch=epochs_total)

            # Save checkpoint after each epoch within curriculum step
            if accelerator.is_main_process and args_log.get('save_checkpoint', True):
                os.makedirs(ckpt_dir, exist_ok=True)
                state_dict = accelerator.get_state_dict(model)
                ckpt_path = os.path.join(ckpt_dir, f'Ct{step}.pt')
                torch.save(state_dict, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\nTraining complete!")
