import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

from source import (
    mnist_transforms_DCGAN,
    show_mnist_data,
    set_seed,
    mnist_transforms_DCGAN_with_normalization,
    DCGANCritic,
    DCGANGenerator,
    GAN,
)

AUGMENTED_MNIST_SHAPE = 32


def main_dcgan(
    data_path: str,
    batch_size: int,
    latent_dim: int,
    normalize_mnist: bool,
    critic_learning_rate: float,
    generator_learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epochs: int,
    output_dir: str,
    save: bool,
    device: str,
    display: bool,
    name_save: str,
    latent_space: str,
    reduced_mnist: float,
    hidden_dim_gen: int,
    hidden_dim_critic :int,
    n_critic_batch :int
):
    logger = logging.getLogger(__name__)
    logger.info("Loading requested data")

    # MNIST dataset, image of size 28x28
    # Resize them to 32x32 (to take the exact same architecture as in paper's experiment on CIFAR-10
    if normalize_mnist:
        transforms = mnist_transforms_DCGAN_with_normalization
    else:
        transforms = mnist_transforms_DCGAN
    train_mnist = MNIST(data_path, train=True, download=True, transform=transforms)

    logger.info("Creating dataloader")
    if reduced_mnist == 0:
        print("Number of images in MNIST train dataset: {}".format(len(train_mnist)))
        train_dataloader = DataLoader(
            train_mnist, batch_size=batch_size, shuffle=True, drop_last=True
        )
    else:
        total_num_in_train_set = len(train_mnist)
        train_size = int((len(train_mnist) * (1 - reduced_mnist)))
        print("Number of bach in train DataLoader: {}".format(train_size))

        train_indices = torch.LongTensor(train_size).random_(0, total_num_in_train_set)
        train_dataloader = torch.utils.data.DataLoader(
            train_mnist,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )

    if display:
        images, labels = next(iter(train_dataloader))
        print("Labels: ", labels)
        print("Batch shape: ", images.size())
        show_mnist_data(images)

    logger.info("Creating models")
    # Models
    output_shape = (1, 32, 32)
    nb_pixel = output_shape[0] * output_shape[1] * output_shape[2]

    critic = DCGANCritic(hidden_dim=hidden_dim_gen).to(device)
    generator = DCGANGenerator(
        latent_dim=latent_dim, hidden_dim=hidden_dim_gen
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

    DCGAN = GAN(
        train_dataloader=train_dataloader,
        latent_dim=latent_dim,
        batch_size=batch_size,
        device=device,
        save=save,
        output_dir=output_dir,
        latent_space=latent_space,
        name_save=name_save,
        optimizer_generator=optimizer_generator,
        optimizer_critic=optimizer_critic,
        generator=generator,
        critic=critic,
        n_critic_batch=n_critic_batch,
    )

    from torch.nn import BCELoss

    # Training
    logger.info("Start training")
    criterion=BCELoss()
    g_losses, c_losses = DCGAN.train(criterion=criterion, epochs=epochs)
    DCGAN.display_image(50, mean=0.1307, sd=0.3081)
    # DCGAN.visualize_generator_outputs()
    # plt.savefig(os.path.join(output_dir, 'generator_output.png'))
    # plt.show()
    plt.plot(g_losses, label='Generator Losses')
    plt.plot(c_losses, label='Critic Losses')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_DCGAN.png'))
    plt.show()
    print('fini')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Set seed")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--data_path", type=str, help="Path to store/retrieve the data")
    parser.add_argument(
        "--normalize_mnist",
        type=bool,
        default=False,
        help="Set to True to normalize MNIST dataset",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--latent_space",
        type=str,
        default="uniform",
        help="Uniform or gaussian latent space",
    )
    parser.add_argument(
        "--reduced_mnist",
        type=float,
        default=0.0,
        help="% redection of mnist dataset",
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs to train models")
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        help="Learning rate for VanillaGANCritic using Adam optimizer",
    )
    parser.add_argument(
        "--generator_learning_rate",
        type=float,
        help="Learning rate for VanillaGANGenerator using Adam optimizer",
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
        "--output_dir", type=str, help="Directory to store best models' checkpoints"
    )
    parser.add_argument(
        "--name_save",
        type=str,
        default="generator_checkpoint.pt",
        help="Name of the file to store/retrieve the data",
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
        "--hidden_dim_gen",
        type=int,
        default=128,
        help="",
    )
    parser.add_argument(
        "--hidden_dim_critic",
        type=int,
        default=1024,
        help="",
    )
    parser.add_argument(
        "--n_critic_batch",
        type=int,
        default=1,
        help="Nombre d'entrainement du critic par batch",
    )

    args = parser.parse_args()

    # set seed
    SEED = args.seed
    set_seed(SEED)

    # potentially change log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main_dcgan(
        data_path=args.data_path,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        normalize_mnist=args.normalize_mnist,
        critic_learning_rate=args.critic_learning_rate,
        generator_learning_rate=args.generator_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epochs=args.epochs,
        output_dir=args.output_dir,
        save=args.save,
        device=args.device,
        display=args.display,
        name_save=args.name_save,
        latent_space=args.latent_space,
        reduced_mnist=args.reduced_mnist,
        hidden_dim_gen=args.hidden_dim_gen,
        hidden_dim_critic=args.hidden_dim_critic,
        n_critic_batch=args.n_critic_batch,

    )
