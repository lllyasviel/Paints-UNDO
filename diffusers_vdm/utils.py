import os
import cv2
import torch
import einops
import torchvision


def resize_and_center_crop(image, target_width, target_height, interpolation=cv2.INTER_AREA):
    original_height, original_width = image.shape[:2]
    k = max(target_height / original_height, target_width / original_width)
    new_width = int(round(original_width * k))
    new_height = int(round(original_height * k))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2
    cropped_image = resized_image[y_start:y_start + target_height, x_start:x_start + target_width]
    return cropped_image


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
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='h264', options={'crf': '1'})
    return x


def save_bcthw_as_png(x, output_filename):
    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, 'b c t h w -> c (b h) (t w)')
    torchvision.io.write_png(x, output_filename)
    return output_filename
