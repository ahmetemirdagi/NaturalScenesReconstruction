import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import noise_like

def extract_into_tensor(a, t, x_shape):
    """
    Helper: gather scalars from a buffer (e.g. alphas_cumprod) into a tensor
    shaped like (batch_size, [1,...,1]).
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMSampler:
    def _init_(self, model, schedule="linear", **kwargs):
        """
        model: your diffusion model instance (e.g. net)
        schedule: not strictly needed for DDPM, but kept for interface compatibility
        """
        super()._init_()
        self.model = model             # e.g. your 'net'
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        # Copy over buffers from your model (betas, alphas_cumprod, etc.)
        # so we can reference them easily below.
        # If your model already has them, you can skip or adapt as needed:
        self.betas = model.betas
        self.alphas_cumprod = model.alphas_cumprod
        self.alphas_cumprod_prev = model.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def make_schedule(self, ddim_num_steps=None, ddim_eta=0., verbose=True):
        """
        Stub function to mimic the DDIM sampler interface.
        In a plain DDPM sampler, we just define which timesteps we will use.
        """
        if ddim_num_steps is None or ddim_num_steps >= self.ddpm_num_timesteps:
            # Use all steps if none provided
            self.timesteps = np.arange(self.ddpm_num_timesteps)
        else:
            # Uniform subsampling of the total steps
            self.timesteps = np.linspace(0, self.ddpm_num_timesteps - 1, ddim_num_steps, dtype=int)

        if verbose:
            print(f"DDPM Sampler: using {len(self.timesteps)} steps (out of {self.ddpm_num_timesteps})")
        # ddpm does not really use eta (DDIM’s parameter), but we keep it for interface compatibility.
        self.ddim_eta = ddim_eta

    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None, use_original_steps=False):
        """
        Forward diffusion noising of x0 at time t. 
        In standard DDPM, x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1_minus_a_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_a_t * x0 + sqrt_1_minus_a_t * noise

    @torch.no_grad()
    def decode_dc(self,
                  x_latent,
                  first_conditioning,
                  second_conditioning,
                  t_start,
                  unconditional_guidance_scale=1.0,
                  xtype='image',
                  first_ctype='vision',
                  second_ctype='prompt',
                  use_original_steps=False,
                  mixed_ratio=0.5,
                  callback=None):
        """
        Reverse diffusion with dual conditioning, from time t_start down to 0.
        """
        # Build the timesteps in descending order.
        timesteps = np.arange(self.ddpm_num_timesteps)[:t_start]
        time_range = np.flip(timesteps)
        total_steps = len(time_range)

        print(f"Running DDPM Decoding with {total_steps} timesteps (dual conditioning)")
        x_dec = x_latent
        for i, step in enumerate(tqdm(time_range, desc='Decoding image', total=total_steps)):
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddpm_dc(
                x_dec,
                first_conditioning,
                second_conditioning,
                ts,
                index=step,
                unconditional_guidance_scale=unconditional_guidance_scale,
                temperature=1.0,
                mixed_ratio=mixed_ratio)
            if callback:
                callback(i)
        return x_dec

    @torch.no_grad()
    def p_sample_ddpm_dc(self, x, first_conditioning, second_conditioning, t, index,
                         unconditional_guidance_scale=1.,
                         temperature=1.,
                         mixed_ratio=0.5,
                         noise_dropout=0.):
        """
        Reverse step with dual conditioning. 
        The model is expected to have something like apply_model_dc(...).
        """
        device = x.device
        b = x.shape[0]

        # Condition input
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        first_c = torch.cat(first_conditioning)
        second_c = torch.cat(second_conditioning)

        # e_t_uncond, e_t => unconditional and conditional noise predictions
        e_t_uncond, e_t = self.model.apply_model_dc(x_in, t_in, first_c, second_c, 
                                                    mixed_ratio=mixed_ratio).chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # Standard DDPM posterior mean computation:
        a_t = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        sqrt_one_minus_at = torch.sqrt(1. - a_t)
        x0_pred = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        a_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        betas_t = extract_into_tensor(self.betas, t, x.shape)

        # Posterior mean
        coeff1 = torch.sqrt(a_prev) * betas_t / (1. - a_t)
        coeff2 = torch.sqrt(a_t) * (1. - a_prev) / (1. - a_t)
        posterior_mean = coeff1 * x0_pred + coeff2 * x

        if index > 0:
            noise = torch.randn_like(x) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            posterior_variance = betas_t * (1. - a_prev) / (1. - a_t)
            x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = posterior_mean
        return x_prev, x0_pred