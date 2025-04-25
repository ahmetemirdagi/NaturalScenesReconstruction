import torch
import numpy as np
from tqdm import tqdm

from .ddpm import DDPMSampler  # Base DDPM sampler


class DDPMSampler_VD(DDPMSampler):
    #def __init__(self, model, num_timesteps_override = None):
     #   super().__init__(model)
      #  if num_timesteps_override is not None:
       #     self.nem_timesteps = num_timesteps_override
        #    self.model.num_timesteps = num_timesteps_override

    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               conditioning=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               xtype='image',
               ctype='prompt',
               temperature=1.,
               verbose=True,
               log_every_t=100):

        print(f'Data shape for DDPM sampling is {shape}')
        bs = shape[0]
        device = self.model.model.diffusion_model.device
        xt = torch.randn(shape, device=device) if xt is None else xt

        intermediates = {'pred_xt': [], 'pred_x0': []}
        pred_xt = xt

        for i in tqdm(reversed(range(self.num_timesteps)), desc='DDPM Sampler', total=self.num_timesteps):
            t = torch.full((bs,), i, device=device, dtype=torch.long)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(pred_xt, t, conditioning, xtype=xtype, ctype=ctype)
            else:
                x_in = torch.cat([pred_xt] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, xtype=xtype, ctype=ctype).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            pred_xt, pred_x0 = self.p_sample(pred_xt, e_t, t)
            if i % log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample(self, x, e_t, t):
        b, *_ = x.shape
        device = x.device

        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(b, *((1,) * (x.ndim - 1)))

        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * e_t) / sqrt_recip_alphas_cumprod_t

        coef1 = self.posterior_mean_coef1.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        coef2 = self.posterior_mean_coef2.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        mean = coef1 * pred_x0 + coef2 * x

        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        var = self.posterior_variance.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        return mean + torch.sqrt(var) * noise, pred_x0

    @torch.no_grad()
    def decode_dc(self, x_latent, first_conditioning, second_conditioning, t_start,
                  unconditional_guidance_scale=1.0, xtype='image',
                  first_ctype='vision', second_ctype='prompt',
                  use_original_steps=False, mixed_ratio=0.5, callback=None):

        time_range = np.flip(np.arange(t_start))
        total_steps = time_range.shape[0]
        print(f"Running DDPM Sampling with {total_steps} timesteps")

        x_dec = x_latent
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            t = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)

            x_dec = self.p_sample_ddpm_dc(
                x_dec,
                first_conditioning,
                second_conditioning,
                t,
                index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                first_ctype=first_ctype,
                second_ctype=second_ctype,
                mixed_ratio=mixed_ratio
            )
            if callback:
                callback(i)

        return x_dec

    @torch.no_grad()
    def p_sample_ddpm_dc(self, x, first_c, second_c, t, index,
                         unconditional_guidance_scale=1.0, xtype='image',
                         first_ctype='vision', second_ctype='prompt', mixed_ratio=0.5):

        b, *_ = x.shape
        device = x.device

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        first_combined = torch.cat(first_c)
        second_combined = torch.cat(second_c)

        e_t_uncond, e_t = self.model.apply_model_dc(
            x_in, t_in, first_combined, second_combined,
            xtype=xtype, first_ctype=first_ctype, second_ctype=second_ctype,
            mixed_ratio=mixed_ratio
        ).chunk(2)

        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * e_t) / sqrt_recip_alphas_cumprod_t

        coef1 = self.posterior_mean_coef1.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        coef2 = self.posterior_mean_coef2.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        mean = coef1 * pred_x0 + coef2 * x

        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        var = self.posterior_variance.to(device)[t].view(b, *((1,) * (x.ndim - 1)))
        return mean + torch.sqrt(var) * noise
    
    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None):
        """
        Add noise to a clean latent x0 at timestep t.
        Used to simulate forward diffusion during DDPM inference.
        """
        device = x0.device
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t].view(x0.shape[0], *((1,) * (x0.ndim - 1)))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(x0.shape[0], *((1,) * (x0.ndim - 1)))

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

