import torchvision.transforms as transforms

# TODO: add normalization ?
mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)