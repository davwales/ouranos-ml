import os
import uuid
import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from RealESRGAN import RealESRGAN
from accelerate.utils import release_memory
from ..base_generator import BaseGenerator

class ImageGenerator(BaseGenerator):
    checkpoint: str
    lora: str
    upscaler_model: str

    model: any
    upscaler: any
    image_generator: any

    def __init__(self, checkpoint: str, lora: str = None, upscaler: str = None):
        self.checkpoint = checkpoint
        self.lora = lora
        self.upscaler_model = upscaler

    def load_model(self):
        pipe = StableDiffusionPipeline.from_single_file(
            self.checkpoint,
            torch_dtype=torch.float16,
            use_safetensors=True)
        
        if self.lora is not None:
            pipe.load_lora_weights(self.lora)
            pipe.fuse_lora()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None
        pipe.to("cuda")
        self.model = pipe

        self.image_generator = torch.Generator("cuda")

        if self.upscaler_model is not None:
            upscaler = RealESRGAN("cuda", scale=2)
            upscaler.load_weights(self.upscaler_model, download=True)
            self.upscaler = upscaler

    def unload_model(self):
        release_memory(self.model)
        release_memory(self.upscaler)
        self.model = None
        self.image_generator = None
        self.upscaler = None

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int, num_inference_steps: int, seed: int, file_name: str):
        if self.image_generator is None or self.model is None:
            raise Exception("Attempting to generate from an unloaded model.")

        if seed is None:
            seed = self.image_generator.seed()
        self.image_generator.manual_seed(seed)

        image = self.model(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=7.0,
            cross_attention_kwargs={"scale": 0.5},
            generator=self.image_generator).images[0]
        
        if self.upscaler is not None:
            image = self.upscaler.predict(image)

        file_name = file_name or f"{int(time.time())}_{uuid.uuid4()}"
        file_ext = "png"
        file_path = os.path.relpath(f"output/images/{file_name}.{file_ext}")
        image.save(file_path, file_ext.upper())
        return (seed, file_path)
