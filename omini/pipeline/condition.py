import torch
from typing import Optional, Union, List, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
from transformers import pipeline
import numpy as np
import cv2
from .tools import encode_images

depth_pipe = None


def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
    blur_radius: Optional[int] = 5,
) -> Union[Image.Image, torch.Tensor]:
    """
    Returns the condition image.
    """
    if condition_type == "depth":
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",  # Use "cpu" to enable parallel processing
        )
        source_image = raw_img.convert("RGB")
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img
    elif condition_type == "canny":
        img = np.array(raw_img)
        edges = cv2.Canny(img, 100, 200)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")
    elif condition_type == "deblurring":
        condition_image = (
            raw_img.convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
        return condition_image
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta=None,
        position_scale=1.0,
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale

    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids, and the adapter name.
        """
        condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        tokens, ids = encode_images(pipe, condition_empty if empty else self.condition)

        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]

        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias

        return tokens, ids
