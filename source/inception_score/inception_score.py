from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

InceptionScoreMetricT = Tuple[torch.Tensor, torch.Tensor]


class MnistScore:
    def __init__(self, classifer_path=None, device=torch.device('cpu'), n_splits: int = 10) -> None:
        torch.manual_seed(42)
        self.n_splits = n_splits
        self.device = device
        if classifer_path is None:
            self.classifier = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(49, 10)
            ).to(self.device)
            self.classifier.load_state_dict(
                torch.load(classifer_path, map_location=device))
        else:
            self.classifier = torch.load(classifer_path, map_location=device)
        self.preprocess = transforms.Compose(
            [
                transforms.CenterCrop((28, 28)),
            ]
        )

    def reshape(self, image):
        image = torch.unsqueeze(image, dim=0)
        return torch.unsqueeze(image, dim=0)

    def get_proba(self, input_image: np.array) -> torch.Tensor:
        self.classifier.eval().requires_grad_(False)
        classif = self.classifier(input_image)
        predictions = torch.softmax(classif, dim=1)
        return (predictions)

    def get_inception_score(self, images: List[np.array]) -> InceptionScoreMetricT:
        images = [self.reshape(self.preprocess(image)) for image in images]
        preds = [self.get_proba(image) for image in images]
        preds = torch.stack(preds)
        print(preds)

        scores = []
        for i in range(self.n_splits):
            part = preds[
                (i * preds.shape[0] // self.n_splits): (
                    (i + 1) * preds.shape[0] // self.n_splits
                ),
                :,
            ]
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0))
            )
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        scores = torch.tensor(scores)
        return torch.mean(scores), torch.std(scores)


class InceptionScore:
    def __init__(self, model: Optional[nn.Module] = None, n_splits: int = 10) -> None:
        torch.manual_seed(42)
        self.n_splits = n_splits
        if model is None:
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
            )
        else:
            self.model = model
        self.model.eval()

        if torch.cuda.is_available():
            self.model.to("cuda")

        # Adapt images to Inception model
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_proba(self, input_image: np.array) -> torch.Tensor:
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities

    def get_inception_score(self, images: List[np.array]) -> InceptionScoreMetricT:
        preds = [self.get_proba(image) for image in images]
        preds = torch.stack(preds)

        scores = []
        for i in range(self.n_splits):
            part = preds[
                (i * preds.shape[0] // self.n_splits): (
                    (i + 1) * preds.shape[0] // self.n_splits
                ),
                :,
            ]
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0))
            )
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        scores = torch.tensor(scores)
        return torch.mean(scores), torch.std(scores)


if __name__ == "__main__":
    from source import generate_stack_images_for_inception_score
    otgan_images = generate_stack_images_for_inception_score(
        generator=otgan_generator,
        latent_dim=50,
        to_rgb=False
    )
    scorer = MnistScore(classifer_path='source/inception_score/classifier.pt')
    mean, std = scorer.get_inception_score(otgan_images)
