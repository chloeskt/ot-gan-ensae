import logging
import os
import random
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from source import Critic, Generator, MinibatchEnergyDistance


def set_seed(seed: int) -> None:
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_ot_gan(
    critic: Critic,
    generator: Generator,
    dataloader: DataLoader,
    optimizer_generator: Optimizer,
    optimizer_critic: Optimizer,
    criterion: MinibatchEnergyDistance,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    n_gen: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: float,
    device: str,
    save: bool,
    output_dir: str,
) -> List[float]:
    # TODO: add EarlyStopping feature
    
    # Instantiate logger
    logger = logging.getLogger(__name__)

    # Make sure models are in train mode
    critic.train()
    generator.train()

    all_losses = []

    epochs_loop = trange(epochs)
    # loop over epochs
    for epoch in epochs_loop:
        running_loss = 0
        batch_loop = tqdm(dataloader, desc="Training of OT-GAN")
        for i, (images, _) in enumerate(batch_loop):
            images.to(device)

            # clear
            optimizer_generator.zero_grad()
            optimizer_critic.zero_grad()

            # sample X, X' from images (real data)
            x, x_prime = torch.split(images, batch_size)

            # generate fake samples from LATENT_DIM dimensional uniform dist between -1 and 1
            z = 2 * torch.rand(batch_size, latent_dim).to(device) - 1
            z_prime = 2 * torch.rand(batch_size, latent_dim).to(device) - 1
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

            if i % (n_gen + 1) == 0:
                # update critic
                loss *= -1
                loss.backward()
                optimizer_critic.step()
            else:
                # update generator
                loss.backward()
                optimizer_generator.step()

            running_loss += loss.item()
            batch_loop.set_postfix({"Loss:": loss.item()})

        # Get average epoch loss
        epoch_loss = running_loss / len(dataloader.dataset)
        all_losses.append(epoch_loss)

        # Add log info
        logger.info(f"[Epoch {epoch}, Loss: {epoch_loss}")

    # Training done, save model if wanted
    if save:
        logger.info(f"Saving models at {output_dir}")
        generator_path = os.path.join(output_dir, "generator_checkpoint.pt")
        critic_path = os.path.join(output_dir, "critic_checkpoint.pt")
        torch.save(generator.state_dict(), generator_path)
        torch.save(critic.state_dict(), critic_path)

    return all_losses


def evaluate_ot_gan():
    raise NotImplementedError
