import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pathlib
import random
import glob
import os


def load_img_cvt_tensor(img_path):
    """
    input : img path
    output : img tensor with shape (1, c, h, w)
    """
    img = Image.open(img_path)
    img = np.asarray(img)/255

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])
    img_tensor = transform(img).type(torch.float)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def load_mulimg_cvt_tensor(imgs_path): 
    """
    input : imgs folder path
    output : img tensor with shape (# of imgs, c, h, w)
    """
    otuput_imgs_tensor = []
    imgs = sorted(glob.glob(os.path.join(imgs_path, "*.jpg")))
    for img_path in imgs:
        img_tensor = load_img_cvt_tensor(img_path)
        otuput_imgs_tensor.append(img_tensor.squeeze(0))

    return torch.stack(otuput_imgs_tensor, 0)

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

def compare_ouptut(imgs_folder1, imgs_folder2):
    imgs1 = sorted(glob.glob(os.path.join(imgs_folder1, "*.jpg")))
    imgs2 = sorted(glob.glob(os.path.join(imgs_folder2, "*.jpg")))

    imgs1 = imgs1[0:min(len(imgs1), len(imgs2))]
    imgs2 = imgs2[0:min(len(imgs1), len(imgs2))]

    for name1, name2 in zip(imgs1, imgs2):
        img1 = np.asarray(Image.open(name1))
        img2 = np.asarray(Image.open(name2))

        if np.any(img1 - img2):
            print('gg')
            print(name1)
            break
    print('good')




if __name__ == '__main__':
    # load_mulimg_cvt_tensor('img/multiple_input')
    compare_ouptut('differentiable_rendering/trail/exp6/image', 'differentiable_rendering/trail/exp5/image')
