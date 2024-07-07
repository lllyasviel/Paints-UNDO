import os
import torch
import einops

from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download
from diffusers_vdm.vae import VideoAutoencoderKL
from diffusers_vdm.projection import Resampler
from diffusers_vdm.unet import UNet3DModel
from diffusers_vdm.improved_clip_vision import ImprovedCLIPVisionModelWithProjection
from diffusers_vdm.dynamic_tsnr_sampler import SamplerDynamicTSNR


class LatentVideoDiffusionPipeline(DiffusionPipeline):
    def __init__(self, tokenizer, text_encoder, image_encoder, vae, image_projection, unet, fp16=True, eval=True):
        super().__init__()

        self.loading_components = dict(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            image_encoder=image_encoder,
            image_projection=image_projection
        )

        for k, v in self.loading_components.items():
            setattr(self, k, v)

        if fp16:
            self.vae.half()
            self.text_encoder.half()
            self.unet.half()
            self.image_encoder.half()
            self.image_projection.half()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()

        if eval:
            self.unet.eval()
            self.image_projection.eval()
        else:
            self.unet.train()
            self.image_projection.train()

    def to(self, *args, **kwargs):
        for k, v in self.loading_components.items():
            if hasattr(v, 'to'):
                v.to(*args, **kwargs)
        return self

    def save_pretrained(self, save_directory, **kwargs):
        for k, v in self.loading_components.items():
            folder = os.path.join(save_directory, k)
            os.makedirs(folder, exist_ok=True)
            v.save_pretrained(folder)
        return

    @classmethod
    def from_pretrained(cls, repo_id, fp16=True, eval=True, token=None):
        local_folder = snapshot_download(repo_id=repo_id, token=token)
        return cls(
            tokenizer=CLIPTokenizer.from_pretrained(os.path.join(local_folder, "tokenizer")),
            text_encoder=CLIPTextModel.from_pretrained(os.path.join(local_folder, "text_encoder")),
            image_encoder=ImprovedCLIPVisionModelWithProjection.from_pretrained(os.path.join(local_folder, "image_encoder")),
            vae=VideoAutoencoderKL.from_pretrained(os.path.join(local_folder, "vae")),
            image_projection=Resampler.from_pretrained(os.path.join(local_folder, "image_projection")),
            unet=UNet3DModel.from_pretrained(os.path.join(local_folder, "unet")),
            fp16=fp16,
            eval=eval
        )

    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        cond_ids = self.tokenizer(prompt,
                                  padding="max_length",
                                  max_length=self.tokenizer.model_max_length,
                                  truncation=True,
                                  return_tensors="pt").input_ids.to(self.text_encoder.device)
        cond = self.text_encoder(cond_ids, attention_mask=None).last_hidden_state
        return cond

    @torch.inference_mode()
    def encode_clip_vision(self, frames):
        b, c, t, h, w = frames.shape
        frames = einops.rearrange(frames, 'b c t h w -> (b t) c h w')
        clipvision_embed = self.image_encoder(frames).last_hidden_state
        clipvision_embed = einops.rearrange(clipvision_embed, '(b t) d c -> b t d c', t=t)
        return clipvision_embed

    @torch.inference_mode()
    def encode_latents(self, videos, return_hidden_states=True):
        b, c, t, h, w = videos.shape
        x = einops.rearrange(videos, 'b c t h w -> (b t) c h w')
        encoder_posterior, hidden_states = self.vae.encode(x, return_hidden_states=return_hidden_states)
        z = encoder_posterior.mode() * self.vae.scale_factor
        z = einops.rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)

        if not return_hidden_states:
            return z

        hidden_states = [einops.rearrange(h, '(b t) c h w -> b c t h w', b=b) for h in hidden_states]
        hidden_states = [h[:, :, [0, -1], :, :] for h in hidden_states]  # only need first and last

        return z, hidden_states

    @torch.inference_mode()
    def decode_latents(self, latents, hidden_states):
        B, C, T, H, W = latents.shape
        latents = einops.rearrange(latents, 'b c t h w -> (b t) c h w')
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype) / self.vae.scale_factor
        pixels = self.vae.decode(latents, ref_context=hidden_states, timesteps=T)
        pixels = einops.rearrange(pixels, '(b t) c h w -> b c t h w', b=B, t=T)
        return pixels

    @torch.inference_mode()
    def __call__(
            self,
            batch_size: int = 1,
            steps: int = 50,
            guidance_scale: float = 5.0,
            positive_text_cond = None,
            negative_text_cond = None,
            positive_image_cond = None,
            negative_image_cond = None,
            concat_cond = None,
            fs = 3,
            progress_tqdm = None,
    ):
        unet_is_training = self.unet.training

        if unet_is_training:
            self.unet.eval()

        device = self.unet.device
        dtype = self.unet.dtype
        dynamic_tsnr_model = SamplerDynamicTSNR(self.unet)

        # Batch

        concat_cond = concat_cond.repeat(batch_size, 1, 1, 1, 1).to(device=device, dtype=dtype)  # b, c, t, h, w
        positive_text_cond = positive_text_cond.repeat(batch_size, 1, 1).to(concat_cond)  # b, f, c
        negative_text_cond = negative_text_cond.repeat(batch_size, 1, 1).to(concat_cond)  # b, f, c
        positive_image_cond = positive_image_cond.repeat(batch_size, 1, 1, 1).to(concat_cond)  # b, t, l, c
        negative_image_cond = negative_image_cond.repeat(batch_size, 1, 1, 1).to(concat_cond)

        if isinstance(fs, torch.Tensor):
            fs = fs.repeat(batch_size, ).to(dtype=torch.long, device=device)  # b
        else:
            fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=device)  # b

        # Initial latents

        latent_shape = concat_cond.shape

        # Feeds

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                context_text=positive_text_cond,
                context_img=positive_image_cond,
                fs=fs,
                concat_cond=concat_cond
            ),
            negative=dict(
                context_text=negative_text_cond,
                context_img=negative_image_cond,
                fs=fs,
                concat_cond=concat_cond
            )
        )

        # Sample

        results = dynamic_tsnr_model(latent_shape, steps, extra_args=sampler_kwargs, progress_tqdm=progress_tqdm)

        if unet_is_training:
            self.unet.train()

        return results
