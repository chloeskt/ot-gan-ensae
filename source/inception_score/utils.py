import os
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from PIL import Image

from ..utils import generate_images_with_generator


def save_images(images, directory="images"):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    i = 0
    for image in images:
        image.save(directory / Path(f"image_{i}.jpeg"))
        i += 1


def grayscale_to_rgb(im: np.array):
    # Tripling the depth
    im = np.repeat(im[..., np.newaxis], 3, -1)
    # float to unit image type (https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type)
    im = (im * 255).astype(np.uint8)
    return im


def generate_stack_images_for_inception_score(
    generator: nn.Module,
    latent_dim: int,
    batch_size: int = 8,
    latent_type: str = "gaussian",
    img_size: int = 32,
    device: str = "cpu",
    to_rgb: bool = True
):
    output = generate_images_with_generator(
        generator=generator,
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        img_size=img_size,
        device=device,
    )
    if to_rgb:
        output = [grayscale_to_rgb(im) for im in output]
        output = [Image.fromarray(im) for im in output]
        save_images(output)
    else:
        output = torch.Tensor(output)
    return output
