# Paints-Undo

PaintsUndo: A Base Model of Drawing Behaviors in Digital Paintings

Paints-Undo is a project aimed at providing base models of human drawing behaviors with a hope that future AI models can better align with the real needs of human artists.

The name "Paints-Undo" is inspired by the similarity that, the model's outputs look like pressing the "undo" button (usually Ctrl+Z) many times in digital painting software.

Paints-Undo presents a family of models that take an image as input and then output the drawing sequence of that image. The model displays all kinds of human behaviors, including but not limited to sketching, inking, coloring, shading, transforming, left-right flipping, color curve tuning, changing the visibility of layers, and even changing the overall idea during the drawing process.

*This page does not contain any examples. All examples are in the below Git page:*

[>>> Click Here to See the Example Page <<<](https://lllyasviel.github.io/pages/paints_undo/)

**This GitHub repo is the only official page of PaintsUndo. We do not have any other websites.**

**Do note that many fake websites of PaintsUndo are on Google and social media recently.**

# Get Started

You can deploy PaintsUndo locally via:

    git clone https://github.com/lllyasviel/Paints-UNDO.git
    cd Paints-UNDO
    conda create -n paints_undo python=3.10
    conda activate paints_undo
    pip install xformers
    pip install -r requirements.txt
    python gradio_app.py

(If you do not know how to use these commands, you can paste those commands to ChatGPT and ask ChatGPT to explain and give more detailed instructions.)

The inference is tested with 24GB VRAM on Nvidia 4090 and 3090TI. It may also work with 16GB VRAM, but does not work with 8GB. My estimation is that, under extreme optimization (including weight offloading and sliced attention), the theoretical minimal VRAM requirement is about 10~12.5 GB.

You can expect to process one image in about 5 to 10 minutes, depending on your settings. As a typical result, you will get a video of 25 seconds at FPS 4, with resolution 320x512, or 512x320, or 384x448, or 448x384.

Because the processing time, in most cases, is significantly longer than most tasks/quota in HuggingFace Space, I personally do not highly recommend to deploy this to HuggingFace Space, to avoid placing an unnecessary burden on the HF servers.

If you do not have required computation devices and still wants an online solution, one option is to wait us to release a Colab notebook (but I am not sure if Colab free tier will work). 

# Model Notes

We currently release two models `paints_undo_single_frame` and `paints_undo_multi_frame`. Let's call them single-frame model and multi-frame model.

The single-frame model takes one image and an `operation step` as input, and outputs one single image. Assuming that an artwork can always be created with 1000 human operations (for example, one brush stroke is one operation), and the `operation step` is an int number from 0 to 999. The number 0 is the finished final artwork, and the number 999 is the first brush stroke drawn on the pure white canvas. You can understand this model as an "undo" (or called Ctrl+Z) model. You input the final image, and indicate how many times you want to "Ctrl+Z", and the model will give you a "simulated" screenshot after those "Ctrl+Z"s are pressed. If your `operation step` is 100, then it means you want to simulate "Ctrl+Z" 100 times on this image to get the appearance after the 100-th "Ctrl+Z".

The multi-frame model takes two images as inputs and output 16 intermediate frames between the two input images. The result is much more consistent than the single-frame model, but also much slower, less "creative", and limited in 16 frames.

In this repo, the default method is to use them together. We will first infer the single-frame model about 5-7 times to get 5-7 "keyframes", and then we use the multi-frame model to "interpolate" those keyframes to actually generate a relatively long video.

In theory this system can be used in many ways and even give infinitely long video, but in practice results are good when the final frame count is about 100-500.

### Model Architecture (paints_undo_single_frame)

The model is a modified architecture of SD1.5 trained on different betas scheduler, clip skip, and the aforementioned `operation step` condition. To be specific, the model is trained with the betas of:

`betas = torch.linspace(0.00085, 0.020, 1000, dtype=torch.float64)`

For comparison, the original SD1.5 is trained with the betas of:

`betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2`

You can notice the difference in the ending betas and the removed square. The choice of this scheduler is based on our internal user study.

The last layer of the text encoder CLIP ViT-L/14 is permanently removed. It is now mathematically consistent to always set CLIP Skip to 2 (if you use diffusers).

The `operation step` condition is added to layer embeddings in a way similar to SDXL's extra embeddings.

Also, since the solo purpose of this model is to process existing images, the model is strictly aligned with WD14 tagger without any other augmentations. You should always use WD14 tagger (the one in this repo) to process the input image to get the prompt. Otherwise, the results may be defective. Human-written prompts are not tested.

### Model Architecture (paints_undo_multi_frame)

This model is trained by resuming from [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) family, but the original Crafter's `lvdm` is not used and all training/inference codes are completely implemented from scratch. (BTW, now the codes are based on modern Diffusers.) Although the initial weights are resumed from VideoCrafter, the topology of neural network is modified a lot, and the network behavior is now largely different from original Crafter after extensive training. 

The overall architecture is like Crafter with 5 components, 3D-UNet, VAE, CLIP, CLIP-Vision, Image Projection.

**VAE**: The VAE is the exactly same anime VAE extracted from [ToonCrafter](https://github.com/ToonCrafter/ToonCrafter). Thanks ToonCrafter a lot for providing the excellent anime temporal VAE for Crafters.

**3D-UNet**: The 3D-UNet is modified from Crafters's `lvdm` with revisions to attention modules. Other than some minor changes in codes, the major change is that now the UNet are trained and supports temporal windows in Spatial Self Attention layers. You can change the codes in `diffusers_vdm.attention.CrossAttention.temporal_window_for_spatial_self_attention` and `temporal_window_type` to activate three types of attention windows:

1. "prv" mode: Each frame's Spatial Self-Attention also attend to full spatial contexts of its previous frame. The first frame only attend itself.
2. "first": Each frame's Spatial Self-Attention also attend to full spatial contexts of the first frame of the entire sequence. The first frame only attend its self.
3. "roll": Each frame's Spatial Self-Attention also attend to full spatial contexts of its previous and next frames, based on the ordering of `torch.roll`.

Note that this is by default disabled in inference to save GPU memory.

**CLIP**: The CLIP of SD2.1.

**CLIP-Vision**: Our implementation of Clip Vision (ViT/H) that supports arbitrary aspect ratios by interpolating the positional embedding. After experimenting with linear interpolation, nearest neighbor, and Rotary Positional Encoding (RoPE), our final choice is nearest neighbor. Note that this is different from Crafter methods that resize or center-crop images to 224x224.

**Image Projection**: Our implementation of a tiny transformer that takes two frames as inputs and outputs 16 image embeddings for each frame. Note that this is different from Crafter methods that only use one image.

# Tutorial

After you get into the Gradio interface:

Step 0: Upload an image or just click an Example image on the bottom of the page.

Step 1: In the UI titled "step 1", click generate prompts to get the global prompt.

Step 2: In the UI titled "step 2", click "Generate Key Frames". You can change seeds or other parameters on the left.

Step 3: In the UI titled "step 3", click "Generate Video". You can change seeds or other parameters on the left.

# Cite

    @Misc{paintsundo,
      author = {Paints-Undo Team},
      title  = {Paints-Undo GitHub Page},
      year   = {2024},
    }

# Applications

Typical use cases of PaintsUndo:

1. Use PaintsUndo as a base model to analyze human behavior to build AI tools that align with human behavior and human demands, for seamless collaboration between AI and humans in a perfectly controlled workflow.

2. Combine PaintsUndo with sketch-guided image generators to achieve “PaintsRedo”, so as to move forward or backward arbitrarily in any of your finished/unfinished artworks to enhance human creativity power. &ast;

3. Use PaintsUndo to view different possible procedures of your own artworks for artistic inspirations.

4. Use the outputs of PaintsUndo as a kind of video/movie After Effects to achieve specific creative purposes.

and much more ...

&ast; *this is already possible - if you use PaintsUndo to Undo 500 steps, and want to Redo 100 steps with different possibilities, you can use ControlNet to finish it (so that it becomes step 0) and then undo 400 steps. More integrated solution is still under experiments.*

# Disclaimer

This project aims to develop base models of human drawing behaviors, facilitating future AI systems to better meet the real needs of human artists. Users are granted the freedom to create content using this tool, but they are expected to comply with local laws and use it responsibly. Users must not employ the tool to generate false information or incite confrontation. The developers do not assume any responsibility for potential misuse by users.
