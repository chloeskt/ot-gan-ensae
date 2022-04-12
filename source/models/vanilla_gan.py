import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Optimizer
from tqdm import trange, tqdm
from early_stopping_pytorch.pytorchtools import EarlyStopping
import logging
import os


# https://github.com/safwankdb/Vanilla-GAN/blob/master/vanilla_gan.py

class VanillaCritic(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, input_size, hidden_dim: int):
        super(VanillaCritic, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(self.layer1.out_features, hidden_dim // 2)
        self.layer3 = nn.Linear(self.layer2.out_features, hidden_dim // 4)
        self.layer4 = nn.Linear(self.layer3.out_features, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.act(self.layer1(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.layer2(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.layer3(x))
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.layer4(x))

        return x  # Shape : (n_batch, 1)


class VanillaGenerator(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape):
        super(VanillaGenerator, self).__init__()

        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(self.layer1.out_features, hidden_dim * 2)
        self.layer3 = nn.Linear(self.layer2.out_features, hidden_dim * 4)
        self.layer4 = nn.Linear(self.layer3.out_features,
                                output_shape[0] * output_shape[1] * output_shape[2])

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = x.view((-1,) + self.output_shape)

        return x  # Shape : (n_batch, 1, 32, 32)


class GAN():
    """GAN implementation (used to train both vanilla GAN and DCGAN)."""

    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 latent_dim: int, batch_size: int, device: str, save: bool,
                 output_dir: str, latent_space: str, name_save: str,
                 optimizer_generator: Optimizer, optimizer_critic: Optimizer,
                 generator, critic, n_gen: int):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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

    def sample_data(self, n_sample):
        if self.latent_space == 'gaussian':
            z_random = torch.randn(n_sample, self.latent_dim).to(self.device)
        else:
            z_random = 2 * torch.rand(n_sample, self.latent_dim).to(self.device) - 1

        samples = self.generator(z_random)
        samples = samples.detach().cpu().numpy()
        return samples

    def train(self, epochs=100, patience=5, criterion=nn.BCELoss()):

        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=self.checkpoint_path
        )
        # Instantiate logger
        logger = logging.getLogger(__name__)

        all_losses = []

        epochs_loop = trange(epochs)
        # loop over epochs
        for epoch in epochs_loop:
            running_loss = 0
            batch_loop = tqdm(self.train_dataloader, desc=f"Epoch {epoch}, Training of GAN")
            for i, (images, _) in enumerate(batch_loop):

                # clear
                self.optimizer_generator.zero_grad()
                self.optimizer_critic.zero_grad()

                # compute loss
                loss = 0
                images = images.to(self.device)

                if self.latent_space == 'gaussian':
                    z_random = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                else:
                    z_random = 2 * torch.rand(self.batch_size, self.latent_dim).to(self.device)

                if i % (self.n_gen + 1) == 0:
                    # update critic
                    #  1A: Train D on real
                    # d_real_data = Variable(batch.cuda())
                    X_real = self.critic(images)
                    y_real = Variable(torch.ones(X_real.shape).to(self.device))
                    d_real_error = criterion(X_real, y_real)

                    #  1B: Train D on fake
                    G_z = self.generator(z_random).detach()  # detach to avoid training G on these labels
                    G = self.critic(G_z)
                    y_fake = Variable(torch.zeros(G.shape).to(self.device))
                    d_fake_error = criterion(G, y_fake)

                    # Backward propagation on the sum of the two losses
                    loss = d_real_error + d_fake_error
                    loss.backward()
                    self.optimizer_critic.step()

                else:
                    # update generator
                    G = self.generator(z_random)
                    G = self.critic(G)
                    y_ones = Variable(torch.ones(G.shape).to(self.device))
                    loss = criterion(G, y_ones)  # Train G to pretend it's genuine

                    loss.backward()
                    self.optimizer_generator.step()

                running_loss += loss.item()

                batch_loop.set_postfix({"Loss:": loss.item()})

            # Get average epoch loss
            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            all_losses.append(epoch_loss)

            running_val_loss = 0
            batch_loop = tqdm(self.val_dataloader, desc="Evaluation of GAN")
            with torch.no_grad():
                for i, (images, _) in enumerate(batch_loop):
                    # clear
                    self.optimizer_generator.zero_grad()
                    self.optimizer_critic.zero_grad()

                    # compute loss
                    images = images.to(self.device)

                    if self.latent_space == 'gaussian':
                        z_random = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                    else:
                        z_random = 2 * torch.rand(self.batch_size, self.latent_dim).to(self.device)

                    G = self.generator(z_random)
                    G = self.critic(G)
                    y_ones = Variable(torch.ones(G.shape).to(self.device))
                    loss = criterion(G, y_ones)
                    batch_loop.set_postfix({"Loss:": loss.item()})

            running_val_loss += loss.item()
            loss=running_val_loss / len(self.val_dataloader.dataset)
            # Early stopping if validation loss increases
            # (only for generator as we update it more often than the critic)
            early_stopping(loss, self.generator)
            if early_stopping.early_stop:
                logger.info("Point of early stopping reached")
                break

            # Add log info
            logger.info(f"Epoch {epoch}, Loss: {epoch_loss}")
            logger.info("\n")

        # load the last checkpoint with the best model
        self.generator.load_state_dict(torch.load(self.checkpoint_path))

        # Training done, save model if wanted
        if self.save:
            logger.info(f"Saving models at {self.output_dir}")
            critic_path = os.path.join(self.output_dir, self.name_save)
            torch.save(self.critic.state_dict(), critic_path)

        return all_losses
# def train_gan_vanilla(
#     critic: VanillaCritic,
#     generator: VanillaGenerator,
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
#     optimizer_generator: Optimizer,
#     optimizer_critic: Optimizer,
#     criterion: nn.BCELoss(),
#     epochs: int,
#     batch_size: int,
#     latent_dim: int,
#     n_gen: int ,
#     patience: int,
#     device: str,
#     save: bool,
#     output_dir: str,
#     latent_space: str,
#     name_save : str,
# ) -> List[float]:
#     # EarlyStopping feature
#     checkpoint_path = os.path.join(output_dir, name_save)
#     early_stopping = EarlyStopping(
#         patience=patience, verbose=True, path=checkpoint_path
#     )
#
#     # Instantiate logger
#     logger = logging.getLogger(__name__)
#
#     # Make sure models are in train mode
#     critic.train()
#     generator.train()
#
#     all_losses = []
#
#     epochs_loop = trange(epochs)
#     # loop over epochs
#     for epoch in epochs_loop:
#         running_loss = 0
#         batch_loop = tqdm(train_dataloader, desc=f"Epoch {epoch}, Training of OT-GAN")
#         for i, (images, _) in enumerate(batch_loop):
#
#             # clear
#             optimizer_generator.zero_grad()
#             optimizer_critic.zero_grad()
#
#             if latent_space == 'gaussian':
#                 z = torch.randn(batch_size, latent_dim).to(device)
#             else:
#                 z = 2 * torch.rand(batch_size, latent_dim).to(device) - 1
#
#             # compute loss
#             loss = criterion()
#
#             if i % (n_gen + 1) == 0:
#                 # update critic
#                 loss *= -1
#
#                 if torch.isnan(loss):
#                     logger.debug("\n")
#                     logger.debug(f"ISSUE WITH LOSS: {loss.item()}")
#                     logger.debug(f"LOSS GRADIENTS: {loss.grad}")
#
#                 loss.backward()
#                 optimizer_critic.step()
#             else:
#                 # update generator
#                 loss.backward()
#                 optimizer_generator.step()
#
#             running_loss += loss.item()
#
#             batch_loop.set_postfix({"Loss:": loss.item()})
#
#         # Get average epoch loss
#         epoch_loss = running_loss / len(train_dataloader.dataset)
#         all_losses.append(epoch_loss)
#
#         # Evaluation and the end of each epoch
#         val_loss = evaluate_ot_gan(
#             critic,
#             generator,
#             val_dataloader,
#             criterion,
#             batch_size,
#             latent_dim,
#             eps_regularization,
#             nb_sinkhorn_iterations,
#             device,
#             latent_space,
#         )
#
#         # Early stopping if validation loss increases
#         # (only for generator as we update it more often than the critic)
#         early_stopping(val_loss, generator)
#         if early_stopping.early_stop:
#             logger.info("Point of early stopping reached")
#             break
#
#         # Add log info
#         logger.info(f"Epoch {epoch}, Loss: {epoch_loss}")
#         logger.info("\n")
#
#     # load the last checkpoint with the best model
#     generator.load_state_dict(torch.load(checkpoint_path))
#
#     # Training done, save model if wanted
#     if save:
#         logger.info(f"Saving models at {output_dir}")
#         critic_path = os.path.join(output_dir, "critic_checkpoint.pt")
#         torch.save(critic.state_dict(), critic_path)
#
#     return all_losses
#
#
# def evaluate_vanilla_gan(
#     critic: Critic,
#     generator: Generator,
#     dataloader: DataLoader,
#     criterion: MinibatchEnergyDistance,
#     batch_size: int,
#     latent_dim: int,
#     eps_regularization: float,
#     nb_sinkhorn_iterations: int,
#     device: str,
#     latent_space : str,
# ) -> float:
#     critic.eval()
#     generator.eval()
#
#     running_val_loss = 0.0
#     batch_loop = tqdm(dataloader, desc="Evaluation of OT-GAN")
#
#     with torch.no_grad():
#         for i, (images, _) in enumerate(batch_loop):
#             images = images.to(device)
#
#             # compute loss, Minibatch Energy Distance
#             loss = process_data_to_retrieve_loss(
#                 images,
#                 generator,
#                 critic,
#                 criterion,
#                 batch_size,
#                 latent_dim,
#                 eps_regularization,
#                 nb_sinkhorn_iterations,
#                 device,
#                 latent_space,
#             )
#             batch_loop.set_postfix({"Loss:": loss.item()})
#         running_val_loss += loss.item()
#
#    return running_val_loss / len(dataloader.dataset)
