import matplotlib.pyplot as plt
import numpy as np
import torchvision


def show_mnist_data(batch_of_images: np.array) -> None:
    im = torchvision.utils.make_grid(batch_of_images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
