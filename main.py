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
    Generator,
    Critic,
    train_ot_gan,
    MinibatchEnergyDistance, set_seed,
)

AUGMENTED_MNIST_SHAPE = 32


def main(
    data_path: str,
    batch_size: int,
    latent_dim: int,
    gen_hidden_dim: int,
    critic_hidden_dim: int,
    nb_channels: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epochs: int,
    n_gen: int,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    output_dir: str,
    save: bool,
    device: str,
    display: bool,
) -> List[float]:
    logger = logging.getLogger(__name__)
    logger.info("Loading requested data")

    # MNIST dataset, image of size 28x28
    # Resize them to 32x32 (to take the exact same architecture as in paper's experiment on CIFAR-10
    mnist = MNIST(data_path, download=True, transform=mnist_transforms)
    print("Number of images in MNIST dataset: {}".format(len(mnist)))

    logger.info("Creating dataloader")
    ot_gan_batch_size = 2 * batch_size
    dataloader = DataLoader(mnist, batch_size=ot_gan_batch_size, shuffle=True)

    if display:
        images, labels = next(iter(dataloader))
        print("Labels: ", labels)
        print("Batch shape: ", images.size())
        show_mnist_data(images)

    logger.info("Creating models")
    # Models
    generator = Generator(
        latent_dim=latent_dim, hidden_dim=gen_hidden_dim, output_dim=nb_channels
    ).to(device)
    critic = Critic(
        output_dim=gen_hidden_dim, hidden_dim=critic_hidden_dim, input_dim=nb_channels
    ).to(device)

    # Check of shapes
    logger.info(f"Summary of the Generator model with input shape ({latent_dim},)")
    summary(generator, input_size=(latent_dim,), device=device)

    logger.info(
        f"Summary of the Critic model with input shape (1, {AUGMENTED_MNIST_SHAPE})"
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
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # LR Schedulers
    # Not implemented for now as none are used in the paper

    logger.info("Instantiate Mini-Batch Energy Distance Loss")
    # Define criterion
    criterion = MinibatchEnergyDistance()

    logger.info("Start training")
    # Training
    train_losses = train_ot_gan(
        critic,
        generator,
        dataloader,
        optimizer_generator,
        optimizer_critic,
        criterion,
        epochs,
        batch_size,
        latent_dim,
        n_gen,
        eps_regularization,
        nb_sinkhorn_iterations,
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
        "--gen_hidden_dim", type=int, default=1024, help="Generator hidden dimension"
    )
    parser.add_argument(
        "--critic_hidden_dim", type=int, default=256, help="Critic hidden dimension"
    )
    parser.add_argument(
        "--nb_output_channels", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs to train models")
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for Adam optimizer"
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

    args = parser.parse_args()

    # set seed
    SEED = args.seed
    set_seed(SEED)

    train_losses = main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        gen_hidden_dim=args.gen_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        nb_channels=args.nb_output_channels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epochs=args.epochs,
        n_gen=args.n_gen,
        eps_regularization=args.eps_regularization,
        nb_sinkhorn_iterations=args.nb_sinkhorn_iterations,
        output_dir=args.output_dir,
        save=args.save,
        device=args.device,
        display=args.display,
    )
