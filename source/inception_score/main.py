import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np

torch.manual_seed(42)
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()

if torch.cuda.is_available():
    model.to('cuda')

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_proba(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
      output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return (probabilities)

def inception_score(images, n_splits=10, eps=1E-16):
    preds = [get_proba(image) for image in images]
    preds = torch.stack(preds)

    scores = []
    for i in range(n_splits):
        part = preds[(i * preds.shape[0] // n_splits):((i + 1) * preds.shape[0] // n_splits), :]
        kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        scores.append(torch.exp(kl))
    scores = torch.tensor(scores)
    return torch.mean(scores), torch.std(scores)

if __name__ == '__main__':
    inception_score([input_image for i in range(20)])
