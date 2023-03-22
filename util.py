import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pathlib
import random


def load_img_cvt_tensor(img_path):
    """
    input : img path
    output : img tensor with shape (1, c, h, w)
    """
    img = Image.open(img_path)
    img = np.asarray(img)/255

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).type(torch.float)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def save_img(idx, output_tensor, workspace):
    to_img = transforms.ToPILImage()
    img = to_img(output_tensor)
    pathlib.Path(f'{workspace}/image/').mkdir(parents=True, exist_ok=True)
    img.save(f'{workspace}/image/{idx}.jpg')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    load_img_cvt_tensor('img/TIME.jpg')
