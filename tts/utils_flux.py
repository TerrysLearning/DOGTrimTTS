import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pipeline_repel.flux_pipeline_repel import FluxNegToMePipelineResample
from pipeline_repel.flux_scheduler_sde import FlowMatchEulerDiscreteSDEScheduler
from pipeline_repel.flux_scheduler import FlowMatchEulerDiscreteODEScheduler
import torch


NTM_ARGS = {
    'merging_alpha': 3.0,
    'merging_threshold': 0.7,
    'merging_t_start': 1000,
    'merging_t_end': 900,
    'merging_dropout': 0.2,
    'push_all': True,
    'num_joint_blocks': -1,
    'num_single_blocks': -1,
}


def load_flux(use_sde=False, device='cuda'):
    if use_sde:
        scheduler = FlowMatchEulerDiscreteSDEScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="scheduler",
        )
    else:
        scheduler = FlowMatchEulerDiscreteODEScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="scheduler",
        )
    pipe = FluxNegToMePipelineResample.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    return pipe


def gen_images(pipe, prompt, latents, start_step=0, end_step=None,
                res=512, num_inference_steps=35, guidance_scale=3.5, batch_size=8,
                use_repel=False, output_type="pil"):

    def _gen_images(prompt, latents, start_step, end_step):
        """Run denoising from start_step to end_step for one batch.
        Returns (noisy_latents, x0) where x0 is either a list of PIL images
        or a packed latent tensor depending on output_type.
        """
        bs = latents.shape[0]
        return pipe(
            prompt=[prompt] * bs,
            latents=latents,
            guidance_scale=guidance_scale,
            height=res,
            width=res,
            num_inference_steps=num_inference_steps,
            generator=None,
            num_images_per_prompt=1,
            use_negtome=use_repel,
            negtome_args=NTM_ARGS if use_repel else None,
            return_dict=False,
            start_step=start_step,
            end_step=end_step,
            output_type=output_type,
        )

    """Partial denoising with batching to respect max_gen_batch_size."""
    with torch.no_grad():
        total = latents.shape[0]
        if total <= batch_size:
            return _gen_images(prompt, latents, start_step, end_step)
        all_lat, all_x0 = [], []
        for i in range(0, total, batch_size):
            lat, x0 = _gen_images(prompt, latents[i:i + batch_size], start_step, end_step)
            all_lat.append(lat)
            all_x0.append(x0)
        result_lat = torch.cat(all_lat, dim=0)
        if output_type == "latent":
            result_x0 = torch.cat(all_x0, dim=0)
        else:
            result_x0 = [img for batch in all_x0 for img in batch]
        del all_lat, all_x0
    return result_lat, result_x0



