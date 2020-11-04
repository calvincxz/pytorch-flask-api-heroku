import json

from commons import get_model, transform_image
import torchvision
import torchvision.transforms as transforms

import requests
from io import BytesIO

net = get_model()
classes = ["Adventure", "Comedy", "Action", "Romance", "Drama", "Crime", "Thriller", "Horror", "Mystery", "Documentary"]

def processImage(img):
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    img_resized = resize(Image.fromarray(img))
    img_as_tensor = to_tensor(img_resized)
    return normalize(img_as_tensor)

def get_prediction(urls):
    images = []
    for idx, url in enumerate(urls):
      response = requests.get(url)
      img = Image.open(BytesIO(response.content))
      images.append(np.array(img))

    npimg = np.array(images)[0]

    with torch.no_grad():
        net.eval()
        processedImage = processImage(npimg)
        output = net(processedImage.view(1, 3, 224, 224).to(device))

    percentages = torch.sigmoid(output[0]).numpy() * 100
    return('Predicted:\n\n' + '\n'.join([(classes[i] + ": %.2f%%" % percentages[i]) for i in range(len(classes))]))
