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

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# print(probabilities)

def get_proba(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
      output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return (probabilities)

def calculate_inception_score(images, n_split=10, eps=1E-16):
    # yhat = images.apply(get_proba)
    yhat = [get_proba(image) for image in images]
    yhat = torch.stack(yhat)
    print('yhat')
    print(yhat.shape)
    scores = list()
    n_batches = np.floor(np.array(len(images))/ n_split)
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = int(i * n_batches), int((i + 1) * n_batches)
        print('ix')
        print(ix_start)
        print(ix_end)
        p_yx = yhat[ix_start:ix_end]
        print('pyx')
        print(p_yx)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        print('py')
        print(p_y)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
# average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

calculate_inception_score([input_image for i in range(20)])
