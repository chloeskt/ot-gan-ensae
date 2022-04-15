from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from IPython.display import display, Image

from .utils import generate_images_with_generator, generate_noise


def show_mnist_data(batch_of_images: np.array) -> None:
    im = torchvision.utils.make_grid(batch_of_images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


def visualize_generator_outputs(
    generator: nn.Module,
    latent_dim: int,
    epoch: Optional[str] = None,
    output_path: Optional[str] = None,
    latent_type: str = "uniform",
    img_size: int = 32,
    batch_size: int = 8,
    save: bool = False,
    device: str = "cpu",
):
    output = generate_images_with_generator(
        generator=generator,
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        img_size=img_size,
        device=device,
    )

    fig = plt.figure(figsize=(batch_size, batch_size))
    if epoch is not None:
        fig.suptitle(f"Generated digits from latent space at epoch {epoch}")
    else:
        fig.suptitle("Generated digits from latent space")

    gridspec = fig.add_gridspec(batch_size, batch_size)
    for idx in range(batch_size**2):
        ax = fig.add_subplot(gridspec[idx])
        ax.imshow(output[idx], cmap="gray")
        ax.set_axis_off()

    if save:
        plt.savefig(output_path)


def create_gif(
    batch_size: int,
    generator: nn.Module,
    latent_dim: int,
    latent_type: str,
    gif_path: str,
    inputs: Optional[np.array] = None,
    img_size: int = 32,
    device: str = "cpu",
    std: float = 1.0,
    mean: float = 0.0,
) -> None:
    if inputs is None:
        inputs = generate_images_with_generator(
            generator=generator,
            batch_size=batch_size,
            latent_dim=latent_dim,
            latent_type=latent_type,
            img_size=img_size,
            device=device,
        )

    # conversion
    inputs = inputs * std + mean
    inputs = inputs * 256
    inputs = inputs.astype(np.uint8)
    # write gif
    imageio.mimwrite(gif_path, inputs, fps=5)


def display_gif(gif_path: str, img_size: int = 200) -> None:
    with open(gif_path, "rb") as f:
        img = Image(data=f.read(), format="png", width=img_size, height=img_size)
        display(img)


def get_interpolation_image(
    generator: nn.Module,
    nb_images: int,
    latent_dim: int,
    latent_type: str,
    device: str,
) -> None:
    start1_noise = generate_noise(
        batch_size=1,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )
    end1_noise = generate_noise(
        batch_size=1,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )
    start2_noise = generate_noise(
        batch_size=1,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )
    end2_noise = generate_noise(
        batch_size=1,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )

    vectors = []
    alphas = torch.linspace(0, 1, int(nb_images**0.5))

    # linear interpolation
    for alpha1 in alphas:
        for alpha2 in alphas:
            v = (
                start1_noise * (1 - alpha1)
                + end1_noise * alpha1
                + start2_noise * (1 - alpha2)
                + end2_noise * alpha2
            ) / 2
            vectors.append(v)

    vectors = torch.vstack(vectors)
    images_generated = generator(vectors).detach().numpy()

    size = int(np.sqrt(nb_images))

    fig, axes = plt.subplots(size, size, figsize=(size, size))
    for ax, img in zip(axes.flatten(), images_generated):
        ax.imshow(img[0], cmap="gray", interpolation="nearest")
        ax.axis("off")
