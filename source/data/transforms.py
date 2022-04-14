import torchvision.transforms as transforms

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))]
)

mnist_transforms_DCGAN = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((28, 28))]
)

mnist_transforms_DCGAN_with_normalization = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((28, 28)),
     transforms.Normalize((0.1307,), (0.3081,))]
)

mnist_transforms_with_normalization = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        #transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
