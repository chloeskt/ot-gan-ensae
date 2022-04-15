import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np

class InceptionScore:
    def __init__(self, n_splits=10):
        torch.manual_seed(42)
        self.n_splits = n_splits
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.eval()

        if torch.cuda.is_available():
            self.model.to('cuda')

        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_proba(self, input_image):
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
          output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return (probabilities)

    def inception_score(self, images):
        preds = [self.get_proba(image) for image in images]
        preds = torch.stack(preds)

        scores = []
        for i in range(self.n_splits):
            part = preds[(i * preds.shape[0] // self.n_splits):((i + 1) * preds.shape[0] // self.n_splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        scores = torch.tensor(scores)
        return torch.mean(scores), torch.std(scores)

if __name__ == '__main__':
    scorer = InceptionScore()
    inception_score = scorer.inception_score([input_image for i in range(20)])
