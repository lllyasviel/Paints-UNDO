import os

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
output_dir = os.path.join('./', 'output')
os.makedirs(output_dir, exist_ok=True)


import functools
import os
import random
import gradio as gr
import numpy as np
import torch
import wd14tagger
import memory_management
import uuid
import time
import json
import warnings
import subprocess

from PIL.ExifTags import TAGS
from PIL import Image, PngImagePlugin
from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_vdm.pipeline import LatentVideoDiffusionPipeline
from diffusers_vdm.utils import resize_and_center_crop, save_bcthw_as_mp4

warnings.filterwarnings("ignore", category=UserWarning)


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
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(torch.bfloat16)
unet = ModifiedUNet.from_pretrained(model_name, subfolder="unet").to(torch.float16)

vae.set_attn_processor(AttnProcessor2_0())
unet.set_attn_processor(AttnProcessor2_0())

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


# Original code by lllyasviel & updated code by MackinationsAi

randomize_symbol = '\U0001F3B2'  # 🎲
folder_symbol = '\U0001F4C2'  # 📂

css = """
#randomize_btn, #randomize_btn_2 {
    margin: 0em 0em 0em 0;
    max-width: 1.5em;
    min-width: 1.5em !important;
    height: 4.35em;
}
#new_project_btn {
    margin-top: 50px;  /* Adjust this value to add more space */
    position: absolute;
    bottom: 0px;
    width: 100%;
}
#theme_btn {
    margin: 0em 0em 0em 0;
    height: 2.5em;
"""

def extract_metadata(image_path):
    if image_path is None:
        return "No image provided."
    
    img = Image.open(image_path)
    metadata = {}
    exif_data = img._getexif()
    
    if exif_data:
        exif_metadata = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            exif_metadata[decoded] = value
        metadata['EXIF'] = exif_metadata

    try:
        png_info = img.info
        if png_info:
            metadata['PNG'] = png_info
    except AttributeError:
        pass

    if metadata:
        return json.dumps(metadata, indent=4)
    else:
        return "No metadata found."

def clear_input():
    return None, ""

def randomize_seed_fn():
    return random.randint(0, 50000)

def create_randomize_button(elem_id):
    randomize_seed_button = gr.Button(randomize_symbol, elem_id=elem_id, variant="primary")
    return randomize_seed_button

utils_folder = "utils"
themes_path = os.path.join(utils_folder, 'themes.json')
config_path = os.path.join(utils_folder, 'config.json')
with open(themes_path, 'r') as f:
    themes = json.load(f)['themes']

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        default_theme = config.get('theme', 'MackinationsAi/dark_evo')
else:
    default_theme = 'MackinationsAi/dark_evo'

def title_ani(name):
    return f"Paints_UNDO, welcome {name}!"

js = f"""
function createGradioAnimation() {{
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.fontFamily = 'monospace';
    container.style.color = '#ffa500';
    container.style.textAlign = 'left';
    container.style.marginBottom = '20px';
    var text = 'Welcome to Paints_UNDO - 🖌️';
    for (var i = 0; i < text.length; i++) {{
        (function(i){{
            setTimeout(function(){{
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.75s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {{
                    letter.style.opacity = '1';
                }}, 50);
            }}, i * 250);
        }})(i);
    }}
    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);
    return 'Animation created';
}}
"""


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

def embed_metadata(image, metadata):
    info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        info.add_text(key, str(value))
    return info

def save_image_with_metadata(image, path, metadata):
    pil_image = Image.fromarray(image)
    info = embed_metadata(pil_image, metadata)
    pil_image.save(path, "PNG", pnginfo=info)

current_project_dir = None

def generate_project_subfolders(force_create=False):
    global current_project_dir
    current_date = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(output_dir, current_date)
    os.makedirs(date_dir, exist_ok=True)

    if current_project_dir is None or force_create:
        project_number = 1
        while os.path.exists(os.path.join(date_dir, f"timelapse_{project_number:04d}")):
            project_number += 1

        current_project_dir = os.path.join(date_dir, f"timelapse_{project_number:04d}")
        os.makedirs(current_project_dir, exist_ok=True)

    input_gpi_dir = os.path.join(current_project_dir, 'input_gpi')
    keyframes_dir = os.path.join(current_project_dir, 'keyframes')
    video_composite_dir = os.path.join(current_project_dir, 'video_composite')
    output_frames_dir = os.path.join(video_composite_dir, 'output_frames')

    os.makedirs(input_gpi_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(video_composite_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    return input_gpi_dir, keyframes_dir, video_composite_dir, output_frames_dir

def create_new_project():
    return generate_project_subfolders(force_create=True)


@torch.inference_mode()
def process(input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg, progress=gr.Progress()):
    rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))

    memory_management.load_models_to_gpu(vae)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    memory_management.load_models_to_gpu(text_encoder)
    conds = encode_cropped_prompt_77tokens(prompt)
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

    input_gpi_dir, keyframes_dir, video_composite_dir, output_frames_dir = generate_project_subfolders()
    key_frame_paths = []
    for i, pixel in enumerate(pixels[1:-1], 1):
        key_frame_metadata = {
            'Prompt': prompt,
            'Negative Prompt': n_prompt,
            'Operation Steps': input_undo_steps,
            '2nd Stage Seed': seed,
            'CFG Scale': cfg,
            'Sample Steps': steps
        }
        key_frame_path = os.path.join(keyframes_dir, f'keyframe_{i:04d}.png')
        save_image_with_metadata(pixel, key_frame_path, key_frame_metadata)
        key_frame_paths.append(key_frame_path)

    return key_frame_paths


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


def extract_frames_from_video(video_path, output_frames_dir, fps):
    os.makedirs(output_frames_dir, exist_ok=True)
    output_frame_template = os.path.join(output_frames_dir, 'output_frames_%04d.png').replace("\\", "/")
    
    command = [
        'ffmpeg',
        '-i', video_path.replace("\\", "/"),
        '-vf', f'fps={fps}',
        output_frame_template
    ]
    
    subprocess.run(command, check=True)

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

    input_gpi_dir, keyframes_dir, video_composite_dir, output_frames_dir = generate_project_subfolders()
    uuid_name = str(uuid.uuid4())
    output_filename = os.path.join(video_composite_dir, uuid_name + '.mp4')

    video = save_bcthw_as_mp4(video, output_filename, fps=fps)

    final_frame_metadata = {
        'Prompt': prompt,
        'Steps': steps,
        'CFG Scale': cfg,
        'FPS': fps,
        'Seed': seed,
    }
    final_frame_image_path = os.path.join(video_composite_dir, uuid_name + '_final_frame.png')
    save_image_with_metadata(cropped_images[-1][1], final_frame_image_path, final_frame_metadata)

    extract_frames_from_video(output_filename, output_frames_dir, fps)

    video = [x.cpu().numpy() for x in video]
    return output_filename, video

def save_initial_frame_with_metadata(prompt, input_fg):
    if input_fg is not None:
        input_gpi_dir, _, _, _ = generate_project_subfolders()
        input_image = Image.fromarray(input_fg)
        initial_metadata = {
            'Prompt': prompt
        }

        existing_files = [f for f in os.listdir(input_gpi_dir) if f.startswith("initial_frame_")]
        next_index = len(existing_files) + 1
        initial_frame_path = os.path.join(input_gpi_dir, f'initial_frame_{next_index:04d}.png')
        
        save_image_with_metadata(np.array(input_image), initial_frame_path, initial_metadata)
    else:
        print("Error: input_fg is None")

def save_theme_to_config(theme):
    with open(config_path, 'w') as f:
        json.dump({'theme': theme}, f)
    return "Theme saved! Please restart the app to apply the new theme."

def create_interface(theme):
    try:
        block = gr.Blocks(theme=theme, css=css, js=js).queue()
        with block:
            with gr.Row(equal_height=True):
                with gr.Column(variant='panel'):
                    gr.HTML(value="<p style='color: #ffa500;'>Digital Painting Timelapse Generator - [<a href='https://github.com/lllyasviel/Paints-UNDO' style='color: #ffa500;'>Github</a>]</p>")
                    with gr.Tabs():
                        with gr.TabItem(label='Upload Image & Generate Prompt - Step 1'):
                            with gr.Row():
                                with gr.Column():
                                    input_fg = gr.Image(sources=['upload'], type="numpy", label="Image", height="512")
                                with gr.Column():
                                    prompt_gen_button = gr.Button(value="Generate Prompt", interactive=True)
                                    prompt = gr.Textbox(label="Output Prompt", interactive=True, lines=3)
                                    with gr.Row():
                                        gr.HTML('<div style="position: relative; height: 100%;">')
                                        new_project_button = gr.Button(value=f"Create New Project - {folder_symbol}", interactive=True, elem_id="new_project_btn")
                                        gr.HTML('</div>')

                        with gr.TabItem(label='Generate KeyFrames - Step 2'):
                            with gr.Row():
                                with gr.Column():
                                    input_undo_steps = gr.Dropdown(label="Operation Steps", value=[1, 100, 300, 500, 700, 900, 950, 999], choices=list(range(1000)), multiselect=True)
                                    with gr.Row():
                                        seed = gr.Slider(label='1st Stage Seed', minimum=0, maximum=50000, step=1, value=12345)
                                        randomize_seed_button = create_randomize_button("randomize_btn")
                                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)
                                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=3.0, step=0.5)
                                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality', lines=1)

                                with gr.Column():
                                    key_gen_button = gr.Button(value="Generate Key Frames", interactive=True)
                                    result_gallery = gr.Gallery(height=512, object_fit='contain', label='Outputs', columns=4)

                        with gr.TabItem(label='Generate All Videos - Step 3'):
                            with gr.Row():
                                with gr.Column():
                                    i2v_input_text = gr.Text(label='Prompts', value='1girl, masterpiece, best quality', lines=9)
                                    with gr.Row():
                                        i2v_seed = gr.Slider(label='2nd Stage Seed', minimum=0, maximum=50000, step=1, value=123)
                                        i2v_randomize_seed_button = create_randomize_button("randomize_btn_2")
                                    i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5, elem_id="i2v_cfg_scale")
                                    i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                                    i2v_fps = gr.Slider(minimum=1, maximum=30, step=1, elem_id="i2v_motion", label="FPS", value=4)
                                with gr.Column():
                                    i2v_end_btn = gr.Button("Generate Video", interactive=True)
                                    i2v_output_video = gr.Video(label="Generated Video", elem_id="output_vid", autoplay=True, show_share_button=True, height=512)
                            with gr.Row():
                                i2v_output_images = gr.Gallery(height=512, label="Output Frames", object_fit="contain", columns=8)

                        with gr.TabItem(label='Metadata Viewer'):
                            gr.HTML(value="<p style='color: #ffa500;'>Upload an image here & its metadata will automatically appear in the DataViewer</p>")
                            with gr.Row():
                                with gr.Column():
                                    image_input = gr.Image(type="filepath", height=717)
                                with gr.Column():
                                    metadata_output = gr.Textbox(label='DataViewer', lines=33)
                                    image_input.change(extract_metadata, inputs=image_input, outputs=metadata_output)
                            with gr.Row():
                                clear_button = gr.Button("Clear Input Image", variant='primary')
                                clear_button.click(clear_input, outputs=[image_input, metadata_output])

                        with gr.TabItem(label='Examples Gallery'):
                            dbs = [
                                ['./imgs/1.jpg', [400, 600, 800, 900, 950, 999], 12345, 512, 640, 50, 3.0, 'lowres, bad anatomy, bad hands, cropped, worst quality', '1girl, masterpiece, best quality', 123, 7.5, 50, 4],
                                ['./imgs/2.jpg', [400, 600, 800, 900, 950, 999], 37000, 512, 640, 50, 3.0, 'lowres, bad anatomy, bad hands, cropped, worst quality', '1girl, masterpiece, best quality', 12345, 7.5, 50, 4],
                                ['./imgs/3.jpg', [400, 600, 800, 900, 950, 999], 3000, 512, 640, 50, 3.0, 'lowres, bad anatomy, bad hands, cropped, worst quality', '1girl, masterpiece, best quality', 3000, 7.5, 50, 4],
                            ]

                            gr.Examples(
                                examples=dbs,
                                inputs=[input_fg, input_undo_steps, seed, image_width, image_height, steps, cfg, n_prompt, i2v_input_text, i2v_seed, i2v_cfg_scale, i2v_steps, i2v_fps],
                                examples_per_page=1024,
                            )

                        with gr.TabItem(label='UI Settings'):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    theme_dropdown = gr.Dropdown(label="Select Theme", choices=themes, value=themes[0])
                                with gr.Column(scale=0.5):
                                    apply_theme_button = gr.Button(value="Save New Theme", interactive=True, elem_id="theme_btn")
                                    restart_message = gr.HTML(value="Click save new theme & then restart app to apply it.", visible=True)
                            
                            apply_theme_button.click(
                                fn=save_theme_to_config,
                                inputs=[theme_dropdown],
                                outputs=[restart_message]
                            )

            input_fg.change(lambda: ["", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)],
                outputs=[prompt, prompt_gen_button, key_gen_button, i2v_end_btn])

            prompt_gen_button.click(
                fn=interrogator_process,
                inputs=[input_fg],
                outputs=[prompt]
            ).then(
                fn=save_initial_frame_with_metadata,
                inputs=[prompt, input_fg],
                outputs=[]
            ).then(
                fn=lambda x: [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)],
                inputs=[],
                outputs=[prompt_gen_button, key_gen_button, i2v_end_btn]
            )

            new_project_button.click(
                fn=create_new_project,
                inputs=[],
                outputs=[]
            )

            key_gen_button.click(
                fn=process,
                inputs=[input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg],
                outputs=[result_gallery]
            )

            i2v_end_btn.click(
                fn=process_video,
                inputs=[result_gallery, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_fps, i2v_seed],
                outputs=[i2v_output_video, i2v_output_images]
            )

            randomize_seed_button.click(
                fn=randomize_seed_fn,
                inputs=[],
                outputs=[seed]
            )

            i2v_randomize_seed_button.click(
                fn=randomize_seed_fn,
                inputs=[],
                outputs=[i2v_seed]
            )

        return block
    except Exception as e:
        print(f"Error in create_interface: {e}")
        return None

block = create_interface(default_theme)
if block is not None:
    block.launch(server_name='0.0.0.0')
else:
    print("Failed to create the Gradio interface.")
