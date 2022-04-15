import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision
from IPython import display

from .utils import generate_images_with_generator

GeneratorT = nn.Module


def show_mnist_data(batch_of_images: np.array) -> None:
    im = torchvision.utils.make_grid(batch_of_images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


def visualize_generator_outputs(
    generator: GeneratorT,
    latent_dim: int,
    latent_type: str = "uniform",
    img_size: int = 32,
    batch_size: int = 8,
):
    output = generate_images_with_generator(
        generator=generator,
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        img_size=img_size,
    )

    fig = plt.figure(figsize=(batch_size, batch_size))
    fig.suptitle("Generated digits from latent space")
    gridspec = fig.add_gridspec(batch_size, batch_size)
    for idx in range(batch_size**2):
        ax = fig.add_subplot(gridspec[idx])
        ax.imshow(output[idx], cmap="gray")
        ax.set_axis_off()


def create_gif(
    batch_size: int,
    generator: GeneratorT,
    latent_dim: int,
    latent_type: str,
    gif_path: str,
    img_size: int = 200,
):
    output = generate_images_with_generator(
        generator=generator,
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        img_size=img_size,
    )

    imageio.mimwrite(gif_path, output, fps=5)

    with open(gif_path, "rb") as f:
        display.Image(data=f.read(), format="png", width=img_size, height=img_size)
