import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from early_stopping_pytorch.pytorchtools import EarlyStopping
from .visualization import create_gif, visualize_generator_outputs
from ..models import OTGANCritic, OTGANGenerator
from ..sinkhorn import MinibatchEnergyDistance


def process_data_to_retrieve_loss(
    images: torch.Tensor,
    generator: OTGANGenerator,
    critic: OTGANCritic,
    criterion: MinibatchEnergyDistance,
    batch_size: int,
    latent_dim: int,
    latent_type: str,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
) -> torch.Tensor:
    # get images on device
    images = images.to(device)

    # sample X, X' from images (real data)
    x, x_prime = torch.split(images, batch_size)

    if latent_type == "uniform":
        # generate fake samples from a latent_dim dimensional uniform dist between -1 and 1
        z = 2 * torch.rand(batch_size, latent_dim).to(device) - 1
        z_prime = 2 * torch.rand(batch_size, latent_dim).to(device) - 1
    else:
        # generate fake samples from a latent_dim dimensinal gaussian dist between 0 and 1
        z = torch.randn(batch_size, latent_dim).to(device)
        z_prime = torch.randn(batch_size, latent_dim).to(device)
    # feed to the generator
    y = generator(z)
    y_prime = generator(z_prime)

    # compute loss, Minibatch Energy Distance
    loss = criterion(
        x,
        x_prime,
        y,
        y_prime,
        critic,
        eps_regularization,
        nb_sinkhorn_iterations,
        device,
    )

    return loss


def train_ot_gan(
    critic: OTGANCritic,
    generator: OTGANGenerator,
    train_dataloader: DataLoader,
    optimizer_generator: Optimizer,
    optimizer_critic: Optimizer,
    criterion: MinibatchEnergyDistance,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    latent_type: str,
    n_gen: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    patience: int,
    device: str,
    save: bool,
    output_dir: str,
    mean: float,
    std: float,
) -> Tuple[List[float], List[float]]:
    # EarlyStopping feature
    checkpoint_path = os.path.join(output_dir, "generator_checkpoint.pt")
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=checkpoint_path
    )

    # Instantiate logger
    logger = logging.getLogger(__name__)

    # Make sure models are in train mode
    critic.train()
    generator.train()

    critic_losses = []
    generator_losses = []

    epochs_loop = trange(epochs)
    # loop over epochs
    for epoch in epochs_loop:
        running_critic_loss = 0
        running_generator_loss = 0
        nb_it_critic = 0
        nb_it_generator = 0
        batch_loop = tqdm(train_dataloader, desc=f"Epoch {epoch}, Training of OT-GAN")
        for i, (images, _) in enumerate(batch_loop):

            # clear
            optimizer_generator.zero_grad()
            optimizer_critic.zero_grad()

            # compute loss
            loss = process_data_to_retrieve_loss(
                images,
                generator,
                critic,
                criterion,
                batch_size,
                latent_dim,
                latent_type,
                eps_regularization,
                nb_sinkhorn_iterations,
                device,
            )

            if i % (n_gen + 1) == 0:
                # update critic
                loss *= -1

                if torch.isnan(loss):
                    print()
                    logger.debug(f"ISSUE WITH LOSS: {loss.item()}")
                    logger.debug(f"LOSS GRADIENTS: {loss.grad}")

                loss.backward()
                optimizer_critic.step()

                running_critic_loss += loss.item()
                nb_it_critic += 1
            else:
                # update generator
                loss.backward()
                optimizer_generator.step()

                running_generator_loss += loss.item()
                nb_it_generator += 1

            batch_loop.set_postfix({"Loss:": loss.item()})

        # Get average epoch loss
        epoch_loss = (running_critic_loss + running_generator_loss) / len(
            train_dataloader.dataset
        )
        critic_losses.append(running_critic_loss / nb_it_critic)
        generator_losses.append(running_generator_loss / nb_it_generator)

        # Early stopping if generator loss increases
        # (only for generator as we update it more often than the critic)
        epoch_generator_loss = running_generator_loss / nb_it_generator
        early_stopping(epoch_generator_loss, generator)
        if early_stopping.early_stop:
            logger.info("Point of early stopping reached")
            break

        # Add log info
        logger.info(f"Epoch {epoch}, Loss: {epoch_loss}")
        print()

        every_x_epoch = 5
        if epoch % (every_x_epoch + 1) == 0:
            output_path = os.path.join(
                output_dir, f"generator_images_epoch_{epoch}.png"
            )
            visualize_generator_outputs(
                generator=generator,
                latent_dim=latent_dim,
                epoch=epoch,
                output_path=output_path,
                save=True,
                device=device,
            )

    # load the last checkpoint with the best model
    generator.load_state_dict(torch.load(checkpoint_path))

    # save generated images for the last epoch done
    output_path = os.path.join(output_dir, f"generator_images_epoch_{epoch}.png")
    visualize_generator_outputs(
        generator=generator,
        latent_dim=latent_dim,
        epoch=epoch,
        output_path=output_path,
        save=True,
        device=device,
    )

    # Training done, save model if wanted
    if save:
        logger.info(f"Saving models at {output_dir}")
        critic_path = os.path.join(output_dir, "critic_checkpoint.pt")
        torch.save(critic.state_dict(), critic_path)

    # make a GIF of images generated by our final generator
    gif_path = os.path.join(output_dir, "final_generator_gif.gif")
    logger.info(f"Creating GIF, saving at {gif_path}")

    create_gif(
        batch_size=8,
        generator=generator,
        latent_dim=latent_dim,
        latent_type=latent_type,
        gif_path=gif_path,
        img_size=32,
        device="cuda",
        mean=mean,
        std=std,
    )

    # TODO:
    # retrieve all previously saved images
    all_paths = []
    for path in os.listdir(output_dir):
        if path.endswith(".png"):
            all_paths.append(os.path.join(output_dir, path))

    # transform them into numpy arrays
    all_np_images = [plt.imread(image) for image in all_paths]

    return critic_losses, generator_losses
