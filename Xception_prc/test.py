import torch
from network.models import model_selection
import cv2
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torch import nn

def main():
    model_path = './pretrained_model/deepfake_c0_xception.pkl'
    img_path = '../Pictures/0023_0.jpg'
    transform1 = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    image = cv2.imread(img_path)
    PIL_image = Image.fromarray(image)
    image = transform1(PIL_image)
    image = Variable(torch.unsqueeze(image, dim=0).cuda(), requires_grad=False)
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    print(preds)
    smax = nn.Softmax(1)
    smax_out = smax(outputs)
    # torch.sum(preds == 0).to(torch.float32)
    print(smax_out)

if __name__ == '__main__':
    main()