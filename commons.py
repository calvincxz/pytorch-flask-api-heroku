import io


from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

def get_model():
    # model = models.densenet121(pretrained=True)
    # model.eval()
    # return model
    ## Load the model based on RESNET18
    PATH = 'model'
    net = torchvision.models.resnet18(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # freeze the layers
    for param in net.parameters():
      param.requires_grad = False

    # Modify the last layer
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(
                          nn.Linear(num_ftrs, 1024), 
                          nn.ReLU(), 
                          nn.Dropout(0.5),
                          nn.Linear(1024, 256), 
                          nn.ReLU(), 
                          nn.Dropout(0.5),
                          nn.Linear(256, len(classes)))

    net.load_state_dict(torch.load(PATH, map_location=str(device)))
    net.to(device)
    return net


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


