import torchvision.transforms as transforms

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)
