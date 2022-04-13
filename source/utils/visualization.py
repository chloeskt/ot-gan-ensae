import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def show_mnist_data(batch_of_images: np.array) -> None:
    im = torchvision.utils.make_grid(batch_of_images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


def visualize_generator_outputs(
    generator,
    latent_dim,
    latent_type="uniform",
    img_size=32,
    batch_size=8,
):
    # Generate fake data
    if latent_type == "uniform":
        z = 2 * torch.rand(batch_size * batch_size, latent_dim) - 1
    else:
        z = torch.randn(batch_size * batch_size, latent_dim)
    output = generator(z).detach()
    output = output.view(-1, 1, img_size, img_size)
    if output.shape[1] == 1:
        output = output.squeeze(dim=1)
    output = np.clip(output, 0, 1)

    fig = plt.figure(figsize=(batch_size, batch_size))
    fig.suptitle("Generated digits from latent space")
    gridspec = fig.add_gridspec(batch_size, batch_size)
    for idx in range(batch_size**2):
        ax = fig.add_subplot(gridspec[idx])
        ax.imshow(output[idx], cmap="gray")
        ax.set_axis_off()
