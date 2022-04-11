import logging
import os
import random
from typing import List

import numpy as np
import torch
from early_stopping_pytorch.pytorchtools import EarlyStopping
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from ..models import Critic, Generator
from ..sinkhorn import MinibatchEnergyDistance


def set_seed(seed: int) -> None:
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def process_data_to_retrieve_loss(
    images: torch.Tensor,
    generator: Generator,
    critic: Critic,
    criterion: MinibatchEnergyDistance,
    batch_size: int,
    latent_dim: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
) -> torch.Tensor:
    # get images on device
    images = images.to(device)

    # sample X, X' from images (real data)
    x, x_prime = torch.split(images, batch_size)

    # generate fake samples from latent_dim dimensional uniform dist between -1 and 1
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

    return loss


def train_ot_gan(
    critic: Critic,
    generator: Generator,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    eval_steps: int,
    optimizer_generator: Optimizer,
    optimizer_critic: Optimizer,
    criterion: MinibatchEnergyDistance,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    n_gen: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    patience: int,
    device: str,
    save: bool,
    output_dir: str,
) -> List[float]:
    # EarlyStopping feature
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=output_dir)

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
        batch_loop = tqdm(train_dataloader, desc="Training of OT-GAN")
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

            # Evaluation every `eval_steps` steps
            if i % (eval_steps + 1) == 0 and i != 0:
                val_loss = evaluate_ot_gan(
                    critic,
                    generator,
                    val_dataloader,
                    criterion,
                    batch_size,
                    latent_dim,
                    eps_regularization,
                    nb_sinkhorn_iterations,
                    device,
                )

                # Early stopping if validation loss increases
                # (only for generator as we update it more often than the critic)
                early_stopping(val_loss, generator)
                if early_stopping.early_stop:
                    logger.info("Point of early stopping reached")
                    break

        # Get average epoch loss
        epoch_loss = running_loss / len(train_dataloader.dataset)
        all_losses.append(epoch_loss)

        # Add log info
        logger.info(f"Epoch {epoch}, Loss: {epoch_loss}")
        logger.info("\n")

    # load the last checkpoint with the best model
    generator.load_state_dict(torch.load("checkpoint.pt"))

    # Training done, save model if wanted
    if save:
        logger.info(f"Saving models at {output_dir}")
        generator_path = os.path.join(output_dir, "generator_checkpoint.pt")
        critic_path = os.path.join(output_dir, "critic_checkpoint.pt")
        torch.save(generator.state_dict(), generator_path)
        torch.save(critic.state_dict(), critic_path)

    return all_losses


def evaluate_ot_gan(
    critic: Critic,
    generator: Generator,
    dataloader: DataLoader,
    criterion: MinibatchEnergyDistance,
    batch_size: int,
    latent_dim: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
) -> float:
    critic.eval()
    generator.eval()

    running_val_loss = 0.0
    batch_loop = tqdm(dataloader, desc="Evaluation of OT-GAN")

    with torch.no_grad():
        for i, (images, _) in enumerate(batch_loop):
            images = images.to(device)

            # compute loss, Minibatch Energy Distance
            loss = process_data_to_retrieve_loss(
                images,
                generator,
                critic,
                criterion,
                batch_size,
                latent_dim,
                eps_regularization,
                nb_sinkhorn_iterations,
                device,
            )
            batch_loop.set_postfix({"Loss:": loss.item()})
        running_val_loss += loss.item()

    return running_val_loss / len(dataloader)
