# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import torch
import torch.nn as nn
import einops

from inspect import isfunction


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(num_groups, channels)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val):
    return val is not None


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_temporal_window(x, t, method):
    assert method in ['roll', 'prv', 'first']

    if method == 'roll':
        m = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
        l = torch.roll(m, shifts=1, dims=1)
        r = torch.roll(m, shifts=-1, dims=1)

        recon = torch.cat([l, m, r], dim=2)
        del l, m, r

        recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
        return recon

    if method == 'prv':
        x = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
        prv = torch.cat([x[:, :1], x[:, :-1]], dim=1)

        recon = torch.cat([x, prv], dim=2)
        del x, prv

        recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
        return recon

    if method == 'first':
        x = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
        prv = x[:, [0], :, :].repeat(1, t, 1, 1)

        recon = torch.cat([x, prv], dim=2)
        del x, prv

        recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
        return recon


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)
