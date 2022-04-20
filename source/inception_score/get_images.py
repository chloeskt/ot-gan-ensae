from source import OTGANGenerator
import torch
from torch import nn
import numpy as np
from pathlib import Path
from PIL import Image
import os

def generate_noise(
    batch_size: int, latent_dim: int, device: str, latent_type: str = "uniform"
) -> torch.Tensor:
    # Generate fake data
    if latent_type == "uniform":
        z = 2 * torch.rand(batch_size * batch_size, latent_dim).to(device) - 1
    elif latent_type == "gaussian":
        z = torch.randn(batch_size * batch_size, latent_dim).to(device)
    else:
        raise NotImplementedError
    return z


def generate_images_with_generator(
    generator: nn.Module,
    batch_size: int,
    latent_dim: int,
    latent_type: str,
    device: str,
    img_size: int = 32,
) -> np.array:
    z = generate_noise(
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )
    output = generator(z).detach().cpu()
    output = output.view(-1, 1, img_size, img_size)
    if output.shape[1] == 1:
        output = output.squeeze(dim=1)
    output = np.clip(output.numpy(), 0, 1)

    return output

def save_images(images, directory="images"):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    i = 0
    for image in images:
        # im = Image.fromarray(image)
        # print("RGB")
        # im = im.convert('RGB')
        image.save(directory/Path(f'image_{i}.jpeg'))
        i += 1

def grayscale_to_rgb(im : np.array):
    im = np.repeat(im[..., np.newaxis], 3, -1) # Tripling the depth
    im = (im * 255).astype(np.uint8) # float to unit image type (https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type)
    return im

def main():
    generator = OTGANGenerator(
            latent_dim=50, hidden_dim=256, kernel_size=3, output_dim=1
        )

    path_to_generator = Path('source/inception_score/generator_checkpoint.pt')
    generator.load_state_dict(torch.load(path_to_generator, map_location=torch.device('cpu')))

    latent_type = 'uniform'
    latent_dim = 50
    img_size = 32
    batch_size = 8
    device = "cpu"
    output = generate_images_with_generator(
            generator=generator,
            batch_size=batch_size,
            latent_dim=latent_dim,
            latent_type=latent_type,
            img_size=img_size,
            device=device,
        )
    output = [grayscale_to_rgb(im) for im in output]
    output = [Image.fromarray(im) for im in output]
    save_images(output)
    return (output)
