# A CLIP Vision supporting arbitrary aspect ratios, by lllyasviel
# The input range is changed to [-1, 1] rather than [0, 1] !!!! (same as VAE's range)

import torch
import types
import einops

from abc import ABCMeta
from transformers import CLIPVisionModelWithProjection


def preprocess(image):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype)[None, :, None, None]

    scale = 16 / min(image.shape[2], image.shape[3])
    image = torch.nn.functional.interpolate(
        image,
        size=(14 * round(scale * image.shape[2]), 14 * round(scale * image.shape[3])),
        mode="bicubic",
        antialias=True
    )

    return (image - mean) / std


def arbitrary_positional_encoding(p, H, W):
    weight = p.weight
    cls = weight[:1]
    pos = weight[1:]
    pos = einops.rearrange(pos, '(H W) C -> 1 C H W', H=16, W=16)
    pos = torch.nn.functional.interpolate(pos, size=(H, W), mode="nearest")
    pos = einops.rearrange(pos, '1 C H W -> (H W) C')
    weight = torch.cat([cls, pos])[None]
    return weight


def improved_clipvision_embedding_forward(self, pixel_values):
    pixel_values = pixel_values * 0.5 + 0.5
    pixel_values = preprocess(pixel_values)
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
    B, C, H, W = patch_embeds.shape
    patch_embeds = einops.rearrange(patch_embeds, 'B C H W -> B (H W) C')
    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + arbitrary_positional_encoding(self.position_embedding, H, W)
    return embeddings


class ImprovedCLIPVisionModelWithProjection(CLIPVisionModelWithProjection, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__(config)
        self.vision_model.embeddings.forward = types.MethodType(
            improved_clipvision_embedding_forward,
            self.vision_model.embeddings
        )
