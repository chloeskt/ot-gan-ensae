import torchvision.transforms as transforms

# MNIST_MEAN = 0.1307
# MNIST_STD = 0.3081

MNIST_MEAN = 0.5
MNIST_STD = 0.5

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)

mnist_transforms_DCGAN = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((28, 28))]
)

mnist_transforms_DCGAN_with_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

mnist_transforms_with_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
)
