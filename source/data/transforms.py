import torchvision.transforms as transforms

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)

mnist_transforms_with_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
