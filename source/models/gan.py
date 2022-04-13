import logging
import os
import imageio
from IPython import display
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from .vanillagan_critic import VanillaGANCritic
from .vanillagan_generator import VanillaGANGenerator


# https://github.com/safwankdb/Vanilla-GAN/blob/master/vanilla_gan.py

# https://github.com/Ksuryateja/DCGAN-MNIST-pytorch/blob/master/gan_mnist.py
# (https://arxiv.org/abs/1511.06434)

LossT = nn.Module

class GAN:
    """GAN implementation (used to train both vanilla GAN and DCGAN)."""

    def __init__(
        self,
        train_dataloader: DataLoader,
        latent_dim: int,
        batch_size: int,
        device: str,
        save: bool,
        output_dir: str,
        latent_space: str,
        name_save: str,
        optimizer_generator: Optimizer,
        optimizer_critic: Optimizer,
        generator: Union[VanillaGANGenerator, None],
        critic: Union[VanillaGANCritic, None],
        n_gen: int,
        gif_name: str = 'gan_gif.gif',
    ):

        self.train_dataloader = train_dataloader
        self.batch_size = batch_size
        self.device = device
        self.save = save
        self.output_dir = output_dir
        self.latent_space = latent_space
        self.name_save = name_save
        self.latent_dim = latent_dim
        self.n_gen = n_gen
        self.optimizer_generator = optimizer_generator
        self.optimizer_critic = optimizer_critic
        self.generator = generator
        self.critic = critic
        self.checkpoint_path = os.path.join(self.output_dir, self.name_save)
        self.gif_name=gif_name
        self.gif_path = os.path.join(self.output_dir, self.gif_name)

    def sample_data(self, n_sample):

        if self.latent_space == "gaussian":
            z_random = torch.randn(n_sample, self.latent_dim).to(self.device)
        else:
            z_random = 2 * torch.rand(n_sample, self.latent_dim).to(self.device) - 1

        samples = self.generator(z_random)
        samples = samples.detach().cpu().numpy()

        return samples

    def display_image(self,n_sample):
        # Generate fake data
        samples=self.sample_data(n_sample)
        samples = samples * 256
        samples = samples.astype(np.uint8)
        samples = np.squeeze(samples, 1)
        imageio.mimwrite(self.gif_path, samples, fps=5)
        gifPath = Path(self.gif_path)
        with open(gifPath, 'rb') as f:
            display.Image(data=f.read(), format='png', width=200, height=200)
        return gifPath

    def visualize_generator_outputs(self, img_size=32, batch_size=8):
        # Generate fake data
        if self.latent_space == "uniform":
            z = 2 * torch.rand(batch_size * batch_size, self.latent_dim) - 1
        else:
            z = torch.randn(batch_size * batch_size, self.latent_dim)
        output = self.generator(z).detach()
        output = output.view(-1, 1, img_size, img_size)
        if output.shape[1] == 1:
            output = output.squeeze(dim=1)
        output = np.clip(output, 0, 1)

        fig = plt.figure(figsize=(batch_size, batch_size))
        fig.suptitle("Generated digits from latent space")
        gridspec = fig.add_gridspec(batch_size, batch_size)
        for idx in range(batch_size ** 2):
            ax = fig.add_subplot(gridspec[idx])
            ax.imshow(output[idx], cmap="gray")
            ax.set_axis_off()


    def make_noise(self):
        if self.latent_space == "gaussian":
            z_random = torch.randn(self.batch_size, self.latent_dim).to(
                self.device
            )
        else:
            z_random = (
                    2 * torch.rand(self.batch_size, self.latent_dim).to(self.device)
                    - 1
            )
        return z_random


    def train(
        self,
        criterion: LossT,
        epochs: int = 100,
        n_gen_batch: int = 1,
        n_critic_batch: int = 1,
    ):

        # Instantiate logger
        logger = logging.getLogger(__name__)

        # Make sure models are in train mode
        self.critic.train()
        self.generator.train()

        loss_generator = []
        loss_critic = []

        epochs_loop = trange(epochs)
        # loop over epochs
        for epoch in epochs_loop:
            running_loss_critic = 0
            running_loss_generator = 0
            batch_loop = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch}, Training of GAN"
            )
            for i, (images, _) in enumerate(batch_loop):

                # clear
                self.optimizer_generator.zero_grad()
                self.optimizer_critic.zero_grad()

                images = images.to(self.device)


                for iter_critic in range(n_critic_batch):
                    # update critic

                    #  1A: Train D on real

                    X_real = self.critic(images)
                    y_real = Variable(torch.ones(X_real.shape).to(self.device))
                    d_real_error = criterion(X_real, y_real)

                    #  1B: Train D on fake

                    G_z = self.generator(
                        self.make_noise()
                    ).detach()  # detach to avoid training G on these labels
                    G = self.critic(G_z)
                    y_fake = Variable(torch.zeros(G.shape).to(self.device))
                    d_fake_error = criterion(G, y_fake)

                    # Backward propagation on the sum of the two losses
                    loss = d_real_error + d_fake_error
                    running_loss_critic += loss.item()
                    loss.backward()
                    self.optimizer_critic.step()

                for iter_gen in range(n_gen_batch):
                    # update generator
                    G = self.generator(self.make_noise())
                    G = self.critic(G)
                    y_ones = Variable(torch.ones(G.shape).to(self.device))
                    loss = criterion(G, y_ones)  # Train G to pretend it's genuine
                    running_loss_generator += loss.item()
                    loss.backward()
                    self.optimizer_generator.step()

                batch_loop.set_postfix({"Loss:": loss.item()})

            # Get average epoch loss
            epoch_loss_critic = running_loss_critic / (
                len(self.train_dataloader.dataset) * n_critic_batch
            )
            loss_critic.append(epoch_loss_critic)
            epoch_loss_generator = running_loss_generator / (
                len(self.train_dataloader.dataset) * n_gen_batch
            )
            loss_generator.append(epoch_loss_generator)

            # Add log info
            logger.info(
                f"Epoch {epoch}, Loss Generator: {epoch_loss_generator}, Loss Critic: {epoch_loss_critic}"
            )
            logger.info("\n")

        # Training done, save model if wanted
        if self.save:
            logger.info(f"Saving generator at {self.output_dir}, as {self.name_save}")
            generator_path = os.path.join(self.output_dir, self.name_save)
            torch.save(self.generator.state_dict(), generator_path)

        return loss_generator, loss_critic