# DOG-Trim 

Arxiv: Rethinking Test Time Scaling for Flow-Matching Generative Models


![Example](assets/dog_knife.png)

## Introducion
![Motivation](assets/motivation_plot.png)

## Setup
```bash
conda create --name tts python=3.12
conda activate tts
pip install transformers -U
pip install git+https://github.com/huggingface/diffusers.git
pip install torch torchvision
pip install matplotlib numba scipy accelerate einops clint ftfy timm fairscale datasets 
pip install git+https://github.com/openai/CLIP.git
```

## Run
Download the narf checkpoint : [todo]
```bash
# input prompt 
python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 10 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt "A dog with a shocked, bug-eyed expression"

# input a json file for all the prompts
# enable multiple GPUs
# use ensemble rewards
python tts/test.py --algorithm global_trim \
    --reward ensemble \
    --n 10 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt "A dog with a shocked, bug-eyed expression"
```

---
### Noise Greedy Search
Start from `n` candidates, pick the best, then search `k` rounds of `m` neighbours within radius `r`.

```bash
python tts/test.py --algorithm bon \
    --reward ensemble \
    --n 6

python tts/test.py --algorithm noise_greedy \
    --reward hpsv2 \
    --n 2 --k 2 --m 2 --r 0.15
```

---

### Noise Epsilon-Greedy Search
Same as noise greedy, but each neighbour is fully random with probability `epsilon`.

```bash
python tts/test.py --algorithm noise_eps_greedy \
    --reward hpsv2 \
    --n 2 --k 2 --m 2 --r 0.15 --epsilon 0.4
```

---

### Search over Path (SoP)
Denoise `d_b` steps, re-noise back `d_f` steps to spawn `m` continuations, keep the best, repeat.

```bash
python tts/test.py --algorithm sop \
    --reward hpsv2 \
    --m 4 --d_b 10 --d_f 5
```

---

### Trajectory Greedy (Local-Path)
Replicate the best latent `n` times at each stage; SDE noise causes divergence. Requires SDE scheduler.

```bash
python tts/test.py --algorithm traj_greedy \
    --reward hpsv2 \
    --n 6 --d 7
```

---

### SMC-SDE
Sequential Monte Carlo with SDE scheduler. Maintain `n` particles, resample by importance weights at each stage.

```bash
python tts/test.py --algorithm smc \
    --reward hpsv2 \
    --n 6 --d 7 \
    --lmbda 50 --tempering increase --potential max --resample ssp

python tts/test.py --algorithm smc \
    --reward pickscore \
    --n 6 --d 7 \
    --lmbda 2 --tempering increase --potential max --resample ssp

python tts/test.py --algorithm smc \
    --reward imagereward \
    --n 6 --d 7 \
    --lmbda 8 --tempering increase --potential max --resample ssp
```
---

### SMC-ODE (Pruning)
Same as SMC-SDE but uses ODE scheduler — resampling becomes pure pruning (unique indices only).

```bash
python tts/test.py --algorithm smc_ode \
    --reward hpsv2 \
    --n 8 --d 7 \
    --lmbda 50 --tempering increase --potential max --resample ssp

python tts/test.py --algorithm smc_ode \
    --reward pickscore \
    --n 8 --d 7 \
    --lmbda 2 --tempering increase --potential max --resample ssp

python tts/test.py --algorithm smc_ode \
    --reward imagereward \
    --n 10 --d 7 \
    --lmbda 8 --tempering increase --potential max --resample ssp

python tts/test.py --algorithm smc_ode \
    --reward ensemble \
    --n 10 --d 7 \
    --lmbda 0.5 --tempering increase --potential max --resample ssp
```

---
### Early Selection
Denoise all `n` candidates to step `d`, select the best, finish denoising to completion.

```bash
python tts/test.py --algorithm early_select \
    --reward hpsv2 \
    --n 13 --d 15
```

---
## Our Method

### Global Trim
Start with `n` candidates. At each stage denoise `d` steps, score, and prune the bottom keeping top `gamma` fraction. Repeat until fully denoised.

```bash
python tts/test.py --algorithm global_trim \
    --reward ensemble \
    --n 15 --d 7 --gamma 0.5 \
    --repel \
    --use_narf
```

With a single reward model:

```bash
python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 15 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json

python tts/test.py --algorithm bon \
    --reward hpsv2 \
    --n 1 \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 15 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json



python tts/test.py --algorithm bon \
    --reward hpsv2 \
    --n 1 \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 5 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 10 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward hpsv2 \
    --n 20 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json



python tts/test.py --algorithm bon \
    --reward imagereward \
    --n 1 \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward imagereward \
    --n 5 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json \
    --num_gpus 2

python tts/test.py --algorithm global_trim \
    --reward imagereward \
    --n 10 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json

python tts/test.py --algorithm global_trim \
    --reward imagereward \
    --n 20 --d 7 --gamma 0.5 --repel --use_narf \
    --prompt assets/interesting.json
```

---

## Notes

- **Ensemble reward** scores with HPSv2 + PickScore + ImageReward and uses rank-sum for search. Individual scores `[hps, pick, imr]` are logged and saved per prompt.
- **SDE scheduler** is loaded automatically for `traj_greedy` and `smc`. All other methods use the ODE scheduler.
- Results are saved to `<out_root>/<run_name>.json` with per-prompt scores and timing. The run resumes automatically if the JSON already exists.




python data/gen_data.py --d 7 --num_prompts 100 --num_gpus 2 --n 20 --prompt_file assets/hpd_train.json