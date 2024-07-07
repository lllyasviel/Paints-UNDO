import os
import json
import random
import glob
import torch
import einops
import torchvision

import safetensors.torch as sf


def write_to_json(data, file_path):
    temp_file_path = file_path + ".tmp"
    with open(temp_file_path, 'wt', encoding='utf-8') as temp_file:
        json.dump(data, temp_file, indent=4)
    os.replace(temp_file_path, file_path)
    return


def read_from_json(file_path):
    with open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_active_parameters(m):
    return {k:v for k, v in m.named_parameters() if v.requires_grad}


def cast_training_params(m, dtype=torch.float32):
    for param in m.parameters():
        if param.requires_grad:
            param.data = param.to(dtype)
    return


def set_attr_recursive(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)
    return


@torch.no_grad()
def batch_mixture(a, b, probability_a=0.5, mask_a=None):
    assert a.shape == b.shape, "Tensors must have the same shape"
    batch_size = a.size(0)

    if mask_a is None:
        mask_a = torch.rand(batch_size) < probability_a

    mask_a = mask_a.to(a.device)
    mask_a = mask_a.reshape((batch_size,) + (1,) * (a.dim() - 1))
    result = torch.where(mask_a, a, b)
    return result


@torch.no_grad()
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def load_last_state(model, folder='accelerator_output'):
    file_pattern = os.path.join(folder, '**', 'model.safetensors')
    files = glob.glob(file_pattern, recursive=True)

    if not files:
        print("No model.safetensors files found in the specified folder.")
        return

    newest_file = max(files, key=os.path.getmtime)
    state_dict = sf.load_file(newest_file)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    print("Loaded model state from:", newest_file)
    return


def generate_random_prompt_from_tags(tags_str, min_length=3, max_length=32):
    tags = tags_str.split(', ')
    tags = random.sample(tags, k=min(random.randint(min_length, max_length), len(tags)))
    prompt = ', '.join(tags)
    return prompt


def save_bcthw_as_mp4(x, output_filename, fps=10):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='h264', options={'crf': '0'})
    return x


def save_bcthw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c t h w -> c (b h) (t w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename


def add_tensors_with_padding(tensor1, tensor2):
    if tensor1.shape == tensor2.shape:
        return tensor1 + tensor2

    shape1 = tensor1.shape
    shape2 = tensor2.shape

    new_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    padded_tensor1 = torch.zeros(new_shape)
    padded_tensor2 = torch.zeros(new_shape)

    padded_tensor1[tuple(slice(0, s) for s in shape1)] = tensor1
    padded_tensor2[tuple(slice(0, s) for s in shape2)] = tensor2

    result = padded_tensor1 + padded_tensor2
    return result
