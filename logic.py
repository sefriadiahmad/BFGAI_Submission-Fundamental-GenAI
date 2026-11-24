import gc
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INPAINT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_text2img_pipeline():
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    return pipe


def generate_basic_image(prompt, negative_prompt="", guidance_scale=7.5, num_inference_steps=30, seed=222):
    device = get_device()
    pipe = load_text2img_pipeline()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        width=512,
        height=512
    )
    return result.images[0]


def load_scheduler(pipe, scheduler_name):
    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate_images(
    prompt,
    negative_prompt="",
    guidance_scale=7.5,
    num_inference_steps=30,
    num_images=1,
    scheduler_name="DPM++",
    seed=222
):
    device = get_device()
    pipe = load_text2img_pipeline()
    pipe = load_scheduler(pipe, scheduler_name)

    generators = [
        torch.Generator(device=device).manual_seed(int(seed) + i)
        for i in range(int(num_images))
    ]

    result = pipe(
        prompt=[prompt] * int(num_images),
        negative_prompt=[negative_prompt] * int(num_images),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generators,
        width=512,
        height=512
    )
    return result.images


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_resource
def load_inpaint_pipeline():
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    return pipe


def inpaint_image(
    image,
    mask,
    prompt,
    negative_prompt="",
    strength=0.95,
    guidance_scale=8.5,
    num_inference_steps=30,
    seed=9
):
    device = get_device()
    pipe_inpaint = load_inpaint_pipeline()

    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = pipe_inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        width=image.width,
        height=image.height
    )
    return result.images[0]


def prepare_outpainting_canvas(image, mode="Zoom Out (All Directions)", direction="Right", expand_pixels=128):
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if mode == "Zoom Out (All Directions)":
        new_w = w + (expand_pixels * 2)
        new_h = h + (expand_pixels * 2)
        paste_x = expand_pixels
        paste_y = expand_pixels
    else:
        if direction in ["Left", "Right"]:
            new_w, new_h = w + expand_pixels, h
        else:
            new_w, new_h = w, h + expand_pixels

        paste_x = expand_pixels if direction == "Left" else 0
        paste_y = expand_pixels if direction == "Top" else 0

    new_w -= new_w % 8
    new_h -= new_h % 8

    bg = image.resize((new_w, new_h), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))
    canvas = bg.copy()
    canvas.paste(image, (paste_x, paste_y))

    mask = Image.new("L", (new_w, new_h), 255)
    inner_box = Image.new("L", (w, h), 0)
    mask.paste(inner_box, (paste_x, paste_y))

    return canvas, mask


def outpaint_image(
    image,
    prompt,
    negative_prompt="",
    mode="Zoom Out (All Directions)",
    direction="Right",
    expand_pixels=128,
    strength=0.95,
    guidance_scale=8.5,
    num_inference_steps=30,
    seed=222
):
    canvas, mask = prepare_outpainting_canvas(
        image=image,
        mode=mode,
        direction=direction,
        expand_pixels=int(expand_pixels)
    )
    output_img = inpaint_image(
        image=canvas,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    return output_img, mask
