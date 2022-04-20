from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

InceptionScoreMetricT = Tuple[torch.Tensor, torch.Tensor]


class MnistScore:
    def __init__(self, device=torch.device('cpu'), n_splits: int = 10) -> None:
        torch.manual_seed(42)
        self.n_splits = n_splits
        self.device = device
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
        self.classifier.load_state_dict('classifier.pt')

    def get_proba(self, input_image: np.array) -> torch.Tensor:
        self.classifier.eval().requires_grad_(False)
        predictions = torch.softmax(classifier(input_image.to(device)), dim=1)
        return (predictions)

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
