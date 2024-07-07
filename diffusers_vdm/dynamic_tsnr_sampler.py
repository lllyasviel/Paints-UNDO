# everything that can improve v-prediction model
# dynamic scaling + tsnr + beta modifier + dynamic cfg rescale + ...
# written by lvmin at stanford 2024

import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from diffusers_vdm.basics import extract_into_tensor


to_torch = partial(torch.tensor, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)

    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)

    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    return noise_cfg


class SamplerDynamicTSNR(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, unet, terminal_scale=0.7):
        super().__init__()
        self.unet = unet

        self.is_v = True
        self.n_timestep = 1000
        self.guidance_rescale = 0.7

        linear_start = 0.00085
        linear_end = 0.012

        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, self.n_timestep, dtype=np.float64) ** 2
        betas = rescale_zero_terminal_snr(betas)
        alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod).to(unet.device))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)).to(unet.device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)).to(unet.device))

        # Dynamic TSNR
        turning_step = 400
        scale_arr = np.concatenate([
            np.linspace(1.0, terminal_scale, turning_step),
            np.full(self.n_timestep - turning_step, terminal_scale)
        ])
        self.register_buffer('scale_arr', to_torch(scale_arr).to(unet.device))

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.sqrt_alphas_cumprod[t] * v + self.sqrt_one_minus_alphas_cumprod[t] * x_t

    def predict_start_from_z_and_v(self, x_t, t, v):
        return self.sqrt_alphas_cumprod[t] * x_t - self.sqrt_one_minus_alphas_cumprod[t] * v

    def q_sample(self, x0, t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def get_v(self, x0, t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * x0)

    def dynamic_x0_rescale(self, x0, t):
        return x0 * extract_into_tensor(self.scale_arr, t, x0.shape)

    @torch.no_grad()
    def get_ground_truth(self, x0, noise, t):
        x0 = self.dynamic_x0_rescale(x0, t)
        xt = self.q_sample(x0, t, noise)
        target = self.get_v(x0, t, noise) if self.is_v else noise
        return xt, target

    def get_uniform_trailing_steps(self, steps):
        c = self.n_timestep / steps
        ddim_timesteps = np.flip(np.round(np.arange(self.n_timestep, 0, -c))).astype(np.int64)
        steps_out = ddim_timesteps - 1
        return torch.tensor(steps_out, device=self.unet.device, dtype=torch.long)

    @torch.no_grad()
    def forward(self, latent_shape, steps, extra_args, progress_tqdm=None):
        bar = tqdm if progress_tqdm is None else progress_tqdm

        eta = 1.0

        timesteps = self.get_uniform_trailing_steps(steps)
        timesteps_prev = torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))

        x = torch.randn(latent_shape, device=self.unet.device, dtype=self.unet.dtype)

        alphas = self.alphas_cumprod[timesteps]
        alphas_prev = self.alphas_cumprod[timesteps_prev]
        scale_arr = self.scale_arr[timesteps]
        scale_arr_prev = self.scale_arr[timesteps_prev]

        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

        s_in = x.new_ones((x.shape[0]))
        s_x = x.new_ones((x.shape[0], ) + (1, ) * (x.ndim - 1))
        for i in bar(range(len(timesteps))):
            index = len(timesteps) - 1 - i
            t = timesteps[index].item()

            model_output = self.model_apply(x, t * s_in, **extra_args)

            if self.is_v:
                e_t = self.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            a_prev = alphas_prev[index].item() * s_x
            sigma_t = sigmas[index].item() * s_x

            if self.is_v:
                pred_x0 = self.predict_start_from_z_and_v(x, t, model_output)
            else:
                a_t = alphas[index].item() * s_x
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # dynamic rescale
            scale_t = scale_arr[index].item() * s_x
            prev_scale_t = scale_arr_prev[index].item() * s_x
            rescale = (prev_scale_t / scale_t)
            pred_x0 = pred_x0 * rescale

            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x

    @torch.no_grad()
    def model_apply(self, x, t, **extra_args):
        x = x.to(device=self.unet.device, dtype=self.unet.dtype)
        cfg_scale = extra_args['cfg_scale']
        p = self.unet(x, t, **extra_args['positive'])
        n = self.unet(x, t, **extra_args['negative'])
        o = n + cfg_scale * (p - n)
        o_better = rescale_noise_cfg(o, p, guidance_rescale=self.guidance_rescale)
        return o_better
