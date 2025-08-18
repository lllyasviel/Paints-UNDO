import os

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
result_dir = os.path.join('./', 'results')
os.makedirs(result_dir, exist_ok=True)


import functools
import os
import random
import gradio as gr
import numpy as np
import torch
import wd14tagger
import memory_management
import uuid

from PIL import Image
from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_vdm.pipeline import LatentVideoDiffusionPipeline
from diffusers_vdm.utils import resize_and_center_crop, save_bcthw_as_mp4


class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m


model_name = 'lllyasviel/paints_undo_single_frame'
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(torch.float16)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(torch.bfloat16)  # bfloat16 vae
unet = ModifiedUNet.from_pretrained(model_name, subfolder="unet").to(torch.float16)

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

video_pipe = LatentVideoDiffusionPipeline.from_pretrained(
    'lllyasviel/paints_undo_multi_frame',
    fp16=True
)

memory_management.unload_all_models([
    video_pipe.unet, video_pipe.vae, video_pipe.text_encoder, video_pipe.image_projection, video_pipe.image_encoder,
    unet, vae, text_encoder
])

k_sampler = KDiffusionSampler(
    unet=unet,
    timesteps=1000,
    linear_start=0.00085,
    linear_end=0.020,
    linear=True
)


def find_best_bucket(h, w, options):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket


@torch.inference_mode()
def encode_cropped_prompt_77tokens(txt: str):
    memory_management.load_models_to_gpu(text_encoder)
    cond_ids = tokenizer(txt,
                         padding="max_length",
                         max_length=tokenizer.model_max_length,
                         truncation=True,
                         return_tensors="pt").input_ids.to(device=text_encoder.device)
    text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond


@torch.inference_mode()
def encode_cropped_prompt(txt: str, max_length=225):
    memory_management.load_models_to_gpu(text_encoder)
    cond_ids = tokenizer(
        txt,
        padding="max_length",
        max_length=max_length + 2,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device=text_encoder.device)
    if max_length + 2 > tokenizer.model_max_length:
        input_ids = cond_ids.squeeze(0)
        id_list = list(range(1, max_length + 2 - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2))
        text_cond_list = []
        for i in id_list:
            # Encode each chunk than concatenate their result
            ids_chunk = (
                input_ids[0].unsqueeze(0),
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),
            )
            if torch.all(ids_chunk[1] == tokenizer.pad_token_id):
                break
            text_cond = text_encoder(torch.concat(ids_chunk).unsqueeze(0)).last_hidden_state
            if text_cond_list == []:
                # BOS token
                text_cond_list.append(text_cond[:, :1])
            text_cond_list.append(text_cond[:, 1:tokenizer.model_max_length - 1])
        # EOS token
        text_cond_list.append(text_cond[:, -1:])
        text_cond = torch.concat(text_cond_list, dim=1)
    else:
        text_cond = text_encoder(
            cond_ids, attention_mask=None
        ).last_hidden_state
    return text_cond.flatten(0, 1).unsqueeze(0)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def interrogator_process(x):
    return wd14tagger.default_interrogator(x)


@torch.inference_mode()
def process(input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg,
            progress=gr.Progress()):
    rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))

    memory_management.load_models_to_gpu(vae)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    memory_management.load_models_to_gpu(text_encoder)
    conds = encode_cropped_prompt(prompt)
    unconds = encode_cropped_prompt_77tokens(n_prompt)

    memory_management.load_models_to_gpu(unet)
    fs = torch.tensor(input_undo_steps).to(device=unet.device, dtype=torch.long)
    initial_latents = torch.zeros_like(concat_conds)
    concat_conds = concat_conds.to(device=unet.device, dtype=unet.dtype)
    latents = k_sampler(
        initial_latent=initial_latents,
        strength=1.0,
        num_inference_steps=steps,
        guidance_scale=cfg,
        batch_size=len(input_undo_steps),
        generator=rng,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        cross_attention_kwargs={'concat_conds': concat_conds, 'coded_conds': fs},
        same_noise_in_batch=True,
        progress_tqdm=functools.partial(progress.tqdm, desc='Generating Key Frames')
    ).to(vae.dtype) / vae.config.scaling_factor

    memory_management.load_models_to_gpu(vae)
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [fg] + pixels + [np.zeros_like(fg) + 255]

    return pixels


@torch.inference_mode()
def process_video_inner(image_1, image_2, prompt, seed=123, steps=25, cfg_scale=7.5, fs=3, progress_tqdm=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    frames = 16

    target_height, target_width = find_best_bucket(
        image_1.shape[0], image_1.shape[1],
        options=[(320, 512), (384, 448), (448, 384), (512, 320)]
    )

    image_1 = resize_and_center_crop(image_1, target_width=target_width, target_height=target_height)
    image_2 = resize_and_center_crop(image_2, target_width=target_width, target_height=target_height)
    input_frames = numpy2pytorch([image_1, image_2])
    input_frames = input_frames.unsqueeze(0).movedim(1, 2)

    memory_management.load_models_to_gpu(video_pipe.text_encoder)
    positive_text_cond = video_pipe.encode_cropped_prompt_77tokens(prompt)
    negative_text_cond = video_pipe.encode_cropped_prompt_77tokens("")

    memory_management.load_models_to_gpu([video_pipe.image_projection, video_pipe.image_encoder])
    input_frames = input_frames.to(device=video_pipe.image_encoder.device, dtype=video_pipe.image_encoder.dtype)
    positive_image_cond = video_pipe.encode_clip_vision(input_frames)
    positive_image_cond = video_pipe.image_projection(positive_image_cond)
    negative_image_cond = video_pipe.encode_clip_vision(torch.zeros_like(input_frames))
    negative_image_cond = video_pipe.image_projection(negative_image_cond)

    memory_management.load_models_to_gpu([video_pipe.vae])
    input_frames = input_frames.to(device=video_pipe.vae.device, dtype=video_pipe.vae.dtype)
    input_frame_latents, vae_hidden_states = video_pipe.encode_latents(input_frames, return_hidden_states=True)
    first_frame = input_frame_latents[:, :, 0]
    last_frame = input_frame_latents[:, :, 1]
    concat_cond = torch.stack([first_frame] + [torch.zeros_like(first_frame)] * (frames - 2) + [last_frame], dim=2)

    memory_management.load_models_to_gpu([video_pipe.unet])
    latents = video_pipe(
        batch_size=1,
        steps=int(steps),
        guidance_scale=cfg_scale,
        positive_text_cond=positive_text_cond,
        negative_text_cond=negative_text_cond,
        positive_image_cond=positive_image_cond,
        negative_image_cond=negative_image_cond,
        concat_cond=concat_cond,
        fs=fs,
        progress_tqdm=progress_tqdm
    )

    memory_management.load_models_to_gpu([video_pipe.vae])
    video = video_pipe.decode_latents(latents, vae_hidden_states)
    return video, image_1, image_2


@torch.inference_mode()
def process_video(keyframes, prompt, steps, cfg, fps, seed, progress=gr.Progress()):
    result_frames = []
    cropped_images = []

    for i, (im1, im2) in enumerate(zip(keyframes[:-1], keyframes[1:])):
        im1 = np.array(Image.open(im1[0]))
        im2 = np.array(Image.open(im2[0]))
        frames, im1, im2 = process_video_inner(
            im1, im2, prompt, seed=seed + i, steps=steps, cfg_scale=cfg, fs=3,
            progress_tqdm=functools.partial(progress.tqdm, desc=f'Generating Videos ({i + 1}/{len(keyframes) - 1})')
        )
        result_frames.append(frames[:, :, :-1, :, :])
        cropped_images.append([im1, im2])

    video = torch.cat(result_frames, dim=2)
    video = torch.flip(video, dims=[2])

    uuid_name = str(uuid.uuid4())
    output_filename = os.path.join(result_dir, uuid_name + '.mp4')
    Image.fromarray(cropped_images[0][0]).save(os.path.join(result_dir, uuid_name + '.png'))
    video = save_bcthw_as_mp4(video, output_filename, fps=fps)
    video = [x.cpu().numpy() for x in video]
    return output_filename, video


block = gr.Blocks().queue()
with block:
    gr.Markdown('# Paints-Undo')

    with gr.Accordion(label='Step 1: Upload Image and Generate Prompt', open=True):
        with gr.Row():
            with gr.Column():
                input_fg = gr.Image(sources=['upload'], type="numpy", label="Image", height=512)
            with gr.Column():
                prompt_gen_button = gr.Button(value="Generate Prompt", interactive=False)
                prompt = gr.Textbox(label="Output Prompt", interactive=True)

    with gr.Accordion(label='Step 2: Generate Key Frames', open=True):
        with gr.Row():
            with gr.Column():
                input_undo_steps = gr.Dropdown(label="Operation Steps", value=[400, 600, 800, 900, 950, 999],
                                               choices=list(range(1000)), multiselect=True)
                seed = gr.Slider(label='Stage 1 Seed', minimum=0, maximum=50000, step=1, value=12345)
                image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=3.0, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality')

            with gr.Column():
                key_gen_button = gr.Button(value="Generate Key Frames", interactive=False)
                result_gallery = gr.Gallery(height=512, object_fit='contain', label='Outputs', columns=4)

    with gr.Accordion(label='Step 3: Generate All Videos', open=True):
        with gr.Row():
            with gr.Column():
                # Note that, at "Step 3: Generate All Videos", using "1girl, masterpiece, best quality"
                # or "1boy, masterpiece, best quality" or just "masterpiece, best quality" leads to better results.
                # Do NOT modify this to use the prompts generated from Step 1 !!
                i2v_input_text = gr.Text(label='Prompts', value='1girl, masterpiece, best quality')
                i2v_seed = gr.Slider(label='Stage 2 Seed', minimum=0, maximum=50000, step=1, value=123)
                i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5,
                                          elem_id="i2v_cfg_scale")
                i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps",
                                      label="Sampling steps", value=50)
                i2v_fps = gr.Slider(minimum=1, maximum=30, step=1, elem_id="i2v_motion", label="FPS", value=4)
            with gr.Column():
                i2v_end_btn = gr.Button("Generate Video", interactive=False)
                i2v_output_video = gr.Video(label="Generated Video", elem_id="output_vid", autoplay=True,
                                            show_share_button=True, height=512)
        with gr.Row():
            i2v_output_images = gr.Gallery(height=512, label="Output Frames", object_fit="contain", columns=8)

    input_fg.change(lambda: ["", gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)],
                    outputs=[prompt, prompt_gen_button, key_gen_button, i2v_end_btn])

    prompt_gen_button.click(
        fn=interrogator_process,
        inputs=[input_fg],
        outputs=[prompt]
    ).then(lambda: [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)],
           outputs=[prompt_gen_button, key_gen_button, i2v_end_btn])

    key_gen_button.click(
        fn=process,
        inputs=[input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg],
        outputs=[result_gallery]
    ).then(lambda: [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)],
           outputs=[prompt_gen_button, key_gen_button, i2v_end_btn])

    i2v_end_btn.click(
        inputs=[result_gallery, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_fps, i2v_seed],
        outputs=[i2v_output_video, i2v_output_images],
        fn=process_video
    )

    dbs = [
        ['./imgs/1.jpg', 12345, 123],
        ['./imgs/2.jpg', 37000, 12345],
        ['./imgs/3.jpg', 3000, 3000],
    ]

    gr.Examples(
        examples=dbs,
        inputs=[input_fg, seed, i2v_seed],
        examples_per_page=1024
    )

block.queue().launch(server_name='0.0.0.0')
