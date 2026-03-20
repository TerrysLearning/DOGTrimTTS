import os
import json
import time
import argparse
import numpy as np
import torch.multiprocessing as mp

from utils_flux import load_flux
from utils_reward import load_reward, wrap_narf
from baselines import noise_search, sop, traj_greedy, smc, early_select
import global_trim

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(description="Test TTS search methods on HPD benchmark")
parser.add_argument('--algorithm',    type=str, required=True,
                    choices=['bon', 'noise_greedy', 'noise_eps_greedy', 'sop', 'traj_greedy', 'smc', 'smc_ode', 'early_select', 'global_trim'])
parser.add_argument('--reward',       type=str, required=True,
                    choices=['pickscore', 'imagereward', 'hpsv2', 'ensemble'])
parser.add_argument('--out_root',     type=str, default='outputs')
parser.add_argument('--prompt',       type=str, required=True,
                    help='path to a .json prompt file, or a single prompt text')
parser.add_argument('--num_gpus',     type=int, default=1,    help='number of GPUs to use in parallel')
parser.add_argument('--seed',         type=int, default=42)
parser.add_argument('--res',          type=int, default=512)
parser.add_argument('--num_inference_steps', type=int, default=35)
parser.add_argument('--gen_batch_size',      type=int, default=8)
parser.add_argument('--reward_batch_size',   type=int, default=16)

# search args
parser.add_argument('--n',       type=int,   default=6,    help='number of initial candidates [bon/noise_greedy/noise_eps_greedy/traj_greedy/smc/smc_ode/early_select/global_trim]')
parser.add_argument('--k',       type=int,   default=2,    help='greedy iterations, 0=BON [noise_greedy/noise_eps_greedy]')
parser.add_argument('--m',       type=int,   default=4,    help='neighbours per greedy iteration [noise_greedy/noise_eps_greedy] / candidate continuations [sop]')
parser.add_argument('--r',       type=float, default=0.15, help='perturbation radius around best noise [noise_greedy/noise_eps_greedy]')
parser.add_argument('--epsilon', type=float, default=0.4,  help='probability of sampling fresh noise instead of perturbing best [noise_eps_greedy]')
parser.add_argument('--d_b', type=int, default=7, help='denoising steps per leg [sop]')
parser.add_argument('--d_f', type=int, default=3, help='re-noising steps per iteration, must be < d_b [sop]')
parser.add_argument('--d',   type=int, default=7, help='denoising steps per stage [traj_greedy/smc/smc_ode/early_select/global_trim]')
parser.add_argument('--lmbda',     type=float, default=50.0,      help='softmax temperature for importance weights [smc/smc_ode]')
parser.add_argument('--tempering', type=str,   default='increase', choices=['increase', 'constant'], help='tempering schedule for scores [smc/smc_ode]')
parser.add_argument('--potential', type=str,   default='max',      choices=['max', 'add', 'diff', 'rt'], help='potential type for importance weights [smc/smc_ode]')
parser.add_argument('--resample',  type=str,   default='ssp',      choices=['ssp', 'multinomial'], help='resampling method [smc/smc_ode]')
parser.add_argument('--gamma',     type=float, default=0.6,        help='fraction of candidates to keep at each pruning stage [global_trim]')

# repel and narf
parser.add_argument('--repel',        action='store_true', default=False, help='enable token repulsion (NegToMe) during generation')
parser.add_argument('--use_narf',     action='store_true', default=False, help='enable noise-aware finetuned reward models')
parser.add_argument('--narf_ckpt_dir', type=str, default='narf/narf_ckpt_v1', help='root dir for NARF checkpoints (expects {dir}/hps/, {dir}/pick/, {dir}/imr/)')


# ============================================================
# Worker — runs in each subprocess, one per GPU
# ============================================================
def worker(rank, args, all_prompts, run_name, out_dir):
    device = f'cuda:{rank}'

    # Per-rank results file to avoid write conflicts
    rank_suffix = f'_rank{rank}' if args.num_gpus > 1 else ''
    results_file = os.path.join(args.out_root, f'{run_name}{rank_suffix}.json')

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        done_pids = {int(k) for k in results if k != '_meta'}
        print(f"[GPU {rank}] Resumed: {len(done_pids)} done")
    else:
        results = {}
        done_pids = set()

    # Load models on this GPU
    sde_algorithms = ['smc', 'traj_greedy']
    use_sde = args.algorithm in sde_algorithms
    pipe         = load_flux(use_sde=use_sde, device=device)
    reward_model = load_reward(args.reward, device=device)
    if args.use_narf:
        reward_model = wrap_narf(reward_model, args.reward, args.narf_ckpt_dir)

    print(f"[GPU {rank}] Ready — handling prompts {rank}, {rank + args.num_gpus}, ...")

    def run_one(prompt):
        common = dict(
            pipe=pipe, reward_model=reward_model, reward_name=args.reward, prompt=prompt,
            seed=args.seed, gen_batch_size=args.gen_batch_size,
            reward_batch_size=args.reward_batch_size,
            res=args.res, num_inference_steps=args.num_inference_steps,
            use_repel=args.repel,
        )
        alg = args.algorithm
        if alg == 'bon':
            return noise_search.search(**common, n=args.n, k=0)
        elif alg == 'noise_greedy':
            return noise_search.search(**common, n=args.n, m=args.m, k=args.k, r=args.r, epsilon=0.0)
        elif alg == 'noise_eps_greedy':
            return noise_search.search(**common, n=args.n, m=args.m, k=args.k, r=args.r, epsilon=args.epsilon)
        elif alg == 'sop':
            return sop.search(**common, m=args.m, d_b=args.d_b, d_f=args.d_f)
        elif alg == 'traj_greedy':
            return traj_greedy.search(**common, n=args.n, d=args.d)
        elif alg == 'smc':
            return smc.search(**common, n=args.n, d=args.d,
                              lmbda=args.lmbda, tempering=args.tempering,
                              potential=args.potential, resample=args.resample)
        elif alg == 'early_select':
            return early_select.search(**common, n=args.n, d=args.d)
        elif alg == 'smc_ode':
            return smc.search(**common, n=args.n, d=args.d,
                              lmbda=args.lmbda, tempering=args.tempering,
                              potential=args.potential, resample=args.resample, ode_case=True)
        elif alg == 'global_trim':
            return global_trim.search(**common, n=args.n, d=args.d, gamma=args.gamma)

    for pid, prompt in enumerate(all_prompts):
        if pid % args.num_gpus != rank:
            continue
        if pid in done_pids:
            continue

        print(f"[GPU {rank}] [{pid:03d}/{len(all_prompts)}] {prompt}")
        t0 = time.time()
        image, score, ensemble_score_info = run_one(prompt)
        elapsed = time.time() - t0

        img_path = os.path.join(out_dir, f"{pid:03d}.png")
        image.save(img_path)

        if ensemble_score_info is not None:
            hps, pick, imr = ensemble_score_info
            results[str(pid)] = {'reward': score, 'hps': hps, 'pick': pick, 'imr': imr, 'time': elapsed}
            print(f"[GPU {rank}]   hps={hps:.4f} pick={pick:.4f} imr={imr:.4f} | time={elapsed:.1f}s")
        else:
            results[str(pid)] = {'reward': score, 'time': elapsed}
            print(f"[GPU {rank}]   reward={score:.4f} | time={elapsed:.1f}s")

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    args = parser.parse_args()

    # Build run name
    if args.algorithm == 'bon':
        run_name = f"images_bon_n{args.n}_{args.reward}"
    elif args.algorithm == 'noise_greedy':
        run_name = f"images_noise_greedy_n{args.n}_k{args.k}_m{args.m}_r{args.r}_{args.reward}"
    elif args.algorithm == 'noise_eps_greedy':
        run_name = f"images_eps_greedy_n{args.n}_k{args.k}_m{args.m}_r{args.r}_eps{args.epsilon}_{args.reward}"
    elif args.algorithm == 'sop':
        run_name = f"images_sop_m{args.m}_db{args.d_b}_df{args.d_f}_{args.reward}"
    elif args.algorithm == 'traj_greedy':
        run_name = f"images_traj_greedy_n{args.n}_d{args.d}_{args.reward}"
    elif args.algorithm == 'early_select':
        run_name = f"images_early_select_n{args.n}_d{args.d}_{args.reward}"
    elif args.algorithm == 'smc':
        run_name = f"images_smc_n{args.n}_d{args.d}_{args.reward}"
    elif args.algorithm == 'smc_ode':
        run_name = f"images_smc_ode_n{args.n}_d{args.d}_{args.reward}"
    elif args.algorithm == 'global_trim':
        run_name = f"images_global_trim_n{args.n}_d{args.d}_gamma{args.gamma}_{args.reward}"

    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.prompt.endswith('.json'):
        with open(args.prompt, 'r') as f:
            all_prompts = json.load(f)
    else:
        all_prompts = [args.prompt]

    print(f"\n{'='*70}")
    print(f"Algorithm : {args.algorithm} | Reward: {args.reward} | run: {run_name}")
    print(f"Out dir   : {out_dir}")
    print(f"GPUs      : {args.num_gpus}")
    print(f"{'='*70}\n")

    total_start = time.time()

    if args.num_gpus == 1:
        worker(0, args, all_prompts, run_name, out_dir)
    else:
        mp.start_method = 'spawn'
        mp.spawn(worker, args=(args, all_prompts, run_name, out_dir), nprocs=args.num_gpus, join=True)

    # ============================================================
    # Merge per-rank results and print summary
    # ============================================================
    results = {}
    for rank in range(args.num_gpus):
        rank_suffix = f'_rank{rank}' if args.num_gpus > 1 else ''
        rf = os.path.join(args.out_root, f'{run_name}{rank_suffix}.json')
        if os.path.exists(rf):
            with open(rf, 'r') as f:
                results.update({k: v for k, v in json.load(f).items() if k != '_meta'})

    total_time = time.time() - total_start
    results['_meta'] = {'total_time': total_time}
    merged_file = os.path.join(args.out_root, f'{run_name}.json')
    with open(merged_file, 'w') as f:
        json.dump(results, f, indent=2)

    all_records = [v for k, v in results.items() if k != '_meta']
    print(f"\n{'='*70}")
    print(f"SUMMARY: {run_name}")
    print(f"  Prompts   : {len(all_records)}")
    if args.reward == 'ensemble':
        hps_scores  = [v['hps']  for v in all_records]
        pick_scores = [v['pick'] for v in all_records]
        imr_scores  = [v['imr']  for v in all_records]
        print(f"  Avg hps   : {np.mean(hps_scores):.4f} ± {np.std(hps_scores):.4f}")
        print(f"  Avg pick  : {np.mean(pick_scores):.4f} ± {np.std(pick_scores):.4f}")
        print(f"  Avg imr   : {np.mean(imr_scores):.4f} ± {np.std(imr_scores):.4f}")
    else:
        all_scores = [v['reward'] for v in all_records]
        print(f"  Avg reward: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*70}")
