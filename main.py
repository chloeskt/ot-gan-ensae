import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import MNIST

from source import mnist_transforms, show_mnist_data, set_seed, Generator, Critic, train_ot_gan, MinibatchEnergyDistance

# Set seed
SEED = 0
set_seed(SEED)

# Define constants
N = 28 * 28
AUGMENTED_MNIST_SHAPE = 32
BATCH_SIZE = 64  # OT-GAN: 2 * batch_size
N_SAMPLES = 18
DATA_PATH = "/mnt/hdd/ot-gan-ensae"
LATENT_DIM = 100
GEN_HIDDEN_DIM = 1024
CRITIC_HIDDEN_DIM = 256
NB_CHANNELS = 1

EPOCHS = 25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
BETA1 = 0.5
BETA2 = 0.999
N_GEN = 3

DEVICE = "cpu"

DISPLAY = False

# MNIST dataset, image of size 28x28
# Resize them to 32x32 (to take the exact same architecture as in paper's experiment on CIFAR-10
# (only a temporary step)
mnist = MNIST(DATA_PATH, download=True, transform=mnist_transforms)
print("Number of images in MNIST dataset: {}".format(len(mnist)))

dataloader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

if DISPLAY:
    images, labels = next(iter(dataloader))
    print("Labels: ", labels)
    print("Batch shape: ", images.size())
    show_mnist_data(images)

# Models
generator = Generator(latent_dim=LATENT_DIM, hidden_dim=GEN_HIDDEN_DIM, output_dim=NB_CHANNELS).to(DEVICE)
critic = Critic(output_dim=GEN_HIDDEN_DIM, hidden_dim=CRITIC_HIDDEN_DIM, input_dim=NB_CHANNELS).to(DEVICE)

# Check of shapes
print(f"Summary of the Generator model with input shape ({LATENT_DIM},)")
summary(generator, input_size=(LATENT_DIM,), device=DEVICE)

print(f"Summary of the Critic model with input shape (1, {AUGMENTED_MNIST_SHAPE})")
summary(critic, input_size=(1, AUGMENTED_MNIST_SHAPE, AUGMENTED_MNIST_SHAPE), device=DEVICE)

# Optimizers
optimizer_generator = torch.optim.Adam(
    generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)
)
optimizer_critic = torch.optim.Adam(
    critic.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)
)

# LR Schedulers
# Not implemented for now as none are used in the paper

# Define criterion
criterion = MinibatchEnergyDistance()

# Training
all_losses = train_ot_gan()
