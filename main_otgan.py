import argparse
import logging
from typing import List

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import MNIST

from source import (
    mnist_transforms,
    show_mnist_data,
    OTGANGenerator,
    OTGANCritic,
    train_ot_gan,
    MinibatchEnergyDistance,
    NewMinibatchEnergyDistance,
    set_seed,
)

AUGMENTED_MNIST_SHAPE = 32


def main(
    data_path: str,
    batch_size: int,
    latent_dim: int,
    latent_type: str,
    kernel_size: int,
    gen_hidden_dim: int,
    critic_hidden_dim: int,
    gen_output_dim: int,
    critic_output_dim: int,
    critic_learning_rate: float,
    generator_learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epochs: int,
    n_gen: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    patience: int,
    output_dir: str,
    save: bool,
    device: str,
    display: bool,
    loss_v0: bool,
) -> List[float]:
    logger = logging.getLogger(__name__)
    logger.info("Loading requested data")

    # MNIST dataset, image of size 28x28
    # Resize them to 32x32 (to take the exact same architecture as in paper's experiment on CIFAR-10
    train_mnist = MNIST(
        data_path, train=True, download=True, transform=mnist_transforms
    )
    print("Number of images in MNIST train dataset: {}".format(len(train_mnist)))

    logger.info("Creating dataloader")
    ot_gan_batch_size = batch_size * 2
    train_dataloader = DataLoader(
        train_mnist, batch_size=ot_gan_batch_size, shuffle=True, drop_last=True
    )

    if display:
        images, labels = next(iter(train_dataloader))
        print("Labels: ", labels)
        print("Batch shape: ", images.size())
        show_mnist_data(images)

    logger.info("Creating models")
    # Models
    generator = OTGANGenerator(
        latent_dim=latent_dim,
        hidden_dim=gen_hidden_dim,
        kernel_size=kernel_size,
        output_dim=gen_output_dim,
    ).to(device)
    critic = OTGANCritic(
        hidden_dim=critic_hidden_dim,
        input_dim=gen_output_dim,
        kernel_size=kernel_size,
        output_dim=critic_output_dim,
    ).to(device)

    # Check of shapes
    logger.info(f"Summary of the OTGANGenerator model with input shape ({latent_dim},)")
    summary(generator, input_size=(latent_dim,), device=device)

    logger.info(
        f"Summary of the OTGANCritic model with input shape (1, {AUGMENTED_MNIST_SHAPE})"
    )
    summary(
        critic,
        input_size=(1, AUGMENTED_MNIST_SHAPE, AUGMENTED_MNIST_SHAPE),
        device=device,
    )

    logger.info("Creating optimizers")
    # Optimizers
    optimizer_generator = torch.optim.Adam(
        generator.parameters(),
        lr=generator_learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=critic_learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # LR Schedulers
    # Not implemented for now as none are used in the paper

    logger.info("Instantiate Mini-Batch Energy Distance Loss")
    # Define criterion
    if loss_v0:
        criterion = MinibatchEnergyDistance()
    else:
        criterion = NewMinibatchEnergyDistance()

    logger.info("Start training")
    # Training
    train_losses = train_ot_gan(
        critic,
        generator,
        train_dataloader,
        optimizer_generator,
        optimizer_critic,
        criterion,
        epochs,
        batch_size,
        latent_dim,
        latent_type,
        n_gen,
        eps_regularization,
        nb_sinkhorn_iterations,
        patience,
        device,
        save,
        output_dir,
    )
    return train_losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Set seed")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--data_path", type=str, help="Path to store/retrieve the data")
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--latent_type",
        type=str,
        default="uniform",
        help="Type of the latent space",
        choices=["gaussian", "uniform"],
    )
    parser.add_argument("--kernel_size", type=int, help="Kernel size", choices=[3, 5])
    parser.add_argument(
        "--gen_hidden_dim",
        type=int,
        default=1024,
        help="OTGANGenerator hidden dimension",
    )
    parser.add_argument(
        "--critic_hidden_dim",
        type=int,
        default=256,
        help="OTGANCritic hidden dimension",
    )
    parser.add_argument(
        "--gen_output_dim",
        type=int,
        default=1,
        help="OTGANGenerator output dimension, should correspond to the number of channels in the image",
    )
    parser.add_argument(
        "--critic_output_dim",
        type=int,
        default=32768,
        help="OTGANCritic output dimension",
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs to train models")
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        help="Learning rate for OTGANCritic using Adam optimizer",
    )
    parser.add_argument(
        "--generator_learning_rate",
        type=float,
        help="Learning rate for OTGANGenerator using Adam optimizer",
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay for Adam optimizer"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--n_gen", type=int, default=3, help="Number of generator steps"
    )
    parser.add_argument(
        "--eps_regularization",
        type=float,
        help="Regularization parameter for Sinkhorn algorithm",
    )
    parser.add_argument(
        "--nb_sinkhorn_iterations",
        type=int,
        help="Number of iterations for Sinkhorn algorithm",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience step for Early Stopping callback - validation loss is monitored",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to store best models' checkpoints"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="Whether to save best models' checkpoints or not",
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Whether to display images, set to True only if you are in a Jupyter notebook",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which to run the code, either cuda or cpu.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Set to True to get DEBUG logs"
    )
    parser.add_argument(
        "--loss_v0",
        type=bool,
        default=True,
        help="Set to True to use MinibatchEnergyDistance and to False to use NewMinibatchEnergyDistance ",
    )

    args = parser.parse_args()

    # set seed
    SEED = args.seed
    set_seed(SEED)

    # potentially change log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    train_losses = main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        latent_type=args.latent_type,
        kernel_size=args.kernel_size,
        gen_hidden_dim=args.gen_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        gen_output_dim=args.gen_output_dim,
        critic_output_dim=args.critic_output_dim,
        critic_learning_rate=args.critic_learning_rate,
        generator_learning_rate=args.generator_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epochs=args.epochs,
        n_gen=args.n_gen,
        eps_regularization=args.eps_regularization,
        nb_sinkhorn_iterations=args.nb_sinkhorn_iterations,
        patience=args.patience,
        output_dir=args.output_dir,
        save=args.save,
        device=args.device,
        display=args.display,
        loss_v0=args.loss_v0,
    )
