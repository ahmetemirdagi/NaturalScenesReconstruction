import torch
import numpy as np
from tqdm import tqdm


class DDPMSampler(object):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_timesteps = model.num_timesteps
        self.device = model.device

        # Precompute posterior coefficients (for DDPM reverse process)
        betas = self.model.betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=betas.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))

        # posterior variance and mean coefficients
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            attr = attr.to(self.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self, batch_size, shape, conditioning=None, x_T=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        C, H, W = shape
        img = torch.randn((batch_size, C, H, W), device=self.device) if x_T is None else x_T

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM Sampling", total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, conditioning, t,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample(self, x, c, t, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        b, *_ = x.shape

        # Conditional or unconditional noise prediction
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)

        # Predict x0
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * e_t) / sqrt_recip_alphas_cumprod_t

        # Compute mean
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * pred_x0 + coef2 * x

        # Add noise, except for the final step
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean + torch.sqrt(var) * noise
