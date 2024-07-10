import tempfile
from pathlib import Path
from contextlib import contextmanager, nullcontext
from time import perf_counter

from tqdm import tqdm
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import numpy as np

from gradio_app import process, process_video, interrogator_process

@contextmanager
def timeof(s: str | None = None):
    duration = []
    start_time = perf_counter()
    yield lambda: duration[0]
    end_time = perf_counter()
    duration.append(end_time - start_time)
    if s is not None:
        print(f"{s}: {duration[0]}")


DUMMY_PROGRESS = type('', (), dict(tqdm=tqdm))

def test_main():
    # Example input from dbs
    input_fg_path = './imgs/1.jpg'
    seed = 12345
    i2v_seed = 123

    # Load the input image
    input_fg = np.array(Image.open(input_fg_path))

    # Generate prompt
    with timeof("Interrogator time") as t_prompt: prompt = interrogator_process(input_fg)
    # prompt = """megumin, 1girl, solo, breasts, looking at viewer, blush, smile, short hair, open mouth, bangs, brown hair, red eyes, long sleeves, dress, bare shoulders, collarbone, upper body, :d, sidelocks, small breasts, choker, indoors, off shoulder, blurry, v-shaped eyebrows, blurry background, black choker, red dress, short hair with long locks, off-shoulder dress"""

    # Generate key frames
    input_undo_steps = [400]#, 600, 800, 900, 950, 999]
    image_width = 512
    image_height = 640
    steps = 50
    cfg = 3.0
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'

    # Create a fake progress object
    with timeof("Processing time") as t_process: key_frames = process(input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg, DUMMY_PROGRESS)

    # Generate video
    i2v_input_text = '1girl, masterpiece, best quality'
    i2v_cfg_scale = 7.5
    i2v_steps = 50
    i2v_fps = 4

    # with nullcontext():
    #     key_frame_paths = [(str(p),) for p in Path('../nottempdir').iterdir()]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir.mkdir(exist_ok=True)
        key_frame_paths = []

        for i, frame in enumerate(key_frames):
            frame_path = Path(temp_dir) / f"frame_{i}.png"
            Image.fromarray(frame).save(frame_path)
            key_frame_paths.append((str(frame_path),))

        print(key_frame_paths)
        with timeof("Video generation time") as t_video:
            output_filename, video = process_video(key_frame_paths, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_fps, i2v_seed, DUMMY_PROGRESS)

    print(f"Generated video saved to: {output_filename}")

# Run the test
test_main()