import torch
import numpy as np

from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger

def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def encode_images(pipeline: FluxPipeline, images: torch.Tensor):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    """
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids
