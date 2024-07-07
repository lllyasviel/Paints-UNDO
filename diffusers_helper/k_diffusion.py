import torch
import numpy as np

from tqdm import tqdm


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, progress_tqdm=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    bar = tqdm if progress_tqdm is None else progress_tqdm

    for i in bar(range(len(sigmas) - 1)):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class KModel:
    def __init__(self, unet, timesteps=1000, linear_start=0.00085, linear_end=0.012, linear=False):
        if linear:
            betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float64)
        else:
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2

        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0
        self.unet = unet
        return

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def __call__(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data ** 2) ** 0.5
        x_ddim_space = x_ddim_space.to(dtype=self.unet.dtype)
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        eps_positive = self.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        eps_negative = self.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        return x - noise_pred * sigma[:, None, None, None]


class KDiffusionSampler:
    def __init__(self, unet, **kwargs):
        self.unet = unet
        self.k_model = KModel(unet=unet, **kwargs)

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent = None,
            strength = 1.0,
            num_inference_steps = 25,
            guidance_scale = 5.0,
            batch_size = 1,
            generator = None,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            cross_attention_kwargs = None,
            same_noise_in_batch = False,
            progress_tqdm = None,
    ):

        device = self.unet.device

        # Sigmas

        sigmas = self.k_model.get_sigmas_karras(int(num_inference_steps/strength))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)

        # Initial latents

        if same_noise_in_batch:
            noise = torch.randn(initial_latent.shape, generator=generator, device=device, dtype=self.unet.dtype).repeat(batch_size, 1, 1, 1)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1).to(device=device, dtype=self.unet.dtype)
        else:
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1).to(device=device, dtype=self.unet.dtype)
            noise = torch.randn(initial_latent.shape, generator=generator, device=device, dtype=self.unet.dtype)

        latents = initial_latent + noise * sigmas[0].to(initial_latent)

        # Batch

        latents = latents.to(device)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1).to(device)

        # Feeds

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        )

        # Sample

        results = sample_dpmpp_2m(self.k_model, latents, sigmas, extra_args=sampler_kwargs, progress_tqdm=progress_tqdm)

        return results
