import math
from tqdm import tqdm
import torch
import torch.nn as nn
from sd import StableDiffusion, seed_everything
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from resize_right import resize

import matplotlib.pyplot as plt
import numpy as np

import tensorboardX

prompt = 'gaming poster'

trail_name = 'exp18'
writer = tensorboardX.SummaryWriter('runs/' + trail_name)

def save_img(idx, output_tensor):
    to_img = transforms.ToPILImage()
    img = to_img(output_tensor)
    # writer.add_image('image', output_tensor, iter)
    img.save('runs/' + trail_name + '/image/' + str(idx)+'.jpg')

bg_content = Image.open('./gaming_bg.jpg')
bg_content = np.asarray(bg_content)/255

img = Image.open('./gaming_title.jpg')
img = np.asarray(img)/255
transform = transforms.Compose([transforms.ToTensor()])
bg_content_tensor = transform(bg_content).type(torch.float)
img_tensor = transform(img).type(torch.float)
bg_content_tensor = bg_content_tensor.unsqueeze(0).to('cuda:0')

print('bg shape', bg_content_tensor.shape)
print('src shape', img_tensor.shape)

src_c, src_h, src_w = img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]
output_w, output_h = 512, 512

src = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]+2*(output_h-img_tensor.shape[1]), img_tensor.shape[2]+2*(output_w-img_tensor.shape[2])))
src[:, output_h-img_tensor.shape[1]:output_h, output_w-img_tensor.shape[2]:output_w] = img_tensor
src = src.unsqueeze(0).to('cuda:0')

#construct mask for combine croped src and bg

mask = torch.ones((img_tensor.shape[0], img_tensor.shape[1]+2*(output_h-img_tensor.shape[1]), img_tensor.shape[2]+2*(output_w-img_tensor.shape[2])))
mask[:, output_h-img_tensor.shape[1]:output_h, output_w-img_tensor.shape[2]:output_w] = 0
mask = mask.unsqueeze(0).to('cuda:0')

#construct linspace grid with size 512*512 at the center
out_y = output_w
out_x = output_h

new_y = torch.linspace(-1/src.shape[2]*output_h, 1/src.shape[2]*output_h, out_y).view(-1,1).repeat(1, out_x)
new_x = torch.linspace(-1/src.shape[3]*output_w, 1/src.shape[3]*output_w, out_x).repeat(out_y, 1)
grid = torch.cat((new_x.unsqueeze(2), new_y.unsqueeze(2)), dim=2)
grid = grid.unsqueeze(0)
grid = grid.to('cuda:0')

#nn parameter
x = nn.Parameter(torch.randn(1, device='cuda:0'))
y = nn.Parameter(torch.randn(1, device='cuda:0'))
w = nn.Parameter(torch.randn(1, device='cuda:0'))
h = nn.Parameter(torch.randn(1, device='cuda:0'))
nn.init.normal_(x)
nn.init.normal_(y)
nn.init.normal_(w)
nn.init.normal_(h)
optimizer = torch.optim.AdamW([x, y, w, h], lr=1e-1, weight_decay=0)

#load sd
device = 'cuda:0'
guidance = StableDiffusion(device)
# prompt = 'fashion magazine cover'
text_embeddings = guidance.get_text_embeds(prompt, '')
guidance.text_encoder.to('cpu')
torch.cuda.empty_cache()

seed_everything(42)

num_steps = 10000
for step in tqdm(range(num_steps)):
    optimizer.zero_grad()

    norm_x = torch.sigmoid(x) * 2 - 1
    norm_y = torch.sigmoid(y) * 2 - 1
    norm_w = torch.sigmoid(w) * 3
    norm_h = torch.sigmoid(h) * 3


    norm_w = norm_w.clamp(min=src_w/output_w, max=10)
    norm_h = norm_h.clamp(min=src_h/output_h, max=10)

    # norm_w = torch.ones(1).to('cuda:0')
    # norm_h = torch.ones(1).to('cuda:0')
    xmin = -2/src.shape[3]*((output_w*norm_w/2)-(src_w/2))
    xmax = 2/src.shape[3]*((output_w*norm_w/2)-(src_w/2))
    ymin = -2/src.shape[2]*((output_h*norm_h/2)-(src_h/2))
    ymax = 2/src.shape[2]*((output_h*norm_h/2)-(src_h/2))

    norm_x = norm_x.clamp(min=xmin, max=xmax)
    norm_y = norm_y.clamp(min=ymin, max=ymax)

    #construct affine transform matrix
    row1 = torch.cat([norm_w, torch.zeros(1).to('cuda:0'), norm_x]).unsqueeze(0)
    row2 = torch.cat([torch.zeros(1).to('cuda:0'), norm_h, norm_y]).unsqueeze(0)
    affine_matrix = torch.cat([row1, row2])

    g = grid.squeeze(0)
    g = g.permute(2,0,1)
    g = g.view(2, output_w*output_h)
    g = torch.cat([g, torch.ones(1, g.shape[1], device='cuda:0')], dim=0)

    g = torch.matmul(affine_matrix, g)
    g = g.view(2, output_h, output_w).permute(1,2,0).unsqueeze(0)



    grid_sample_src = F.grid_sample(src, grid=g, mode='bilinear')
    grid_sample_mask = F.grid_sample(mask, grid=g, mode='bilinear', padding_mode="border")
    output = bg_content_tensor * grid_sample_mask + grid_sample_src 

    t = torch.squeeze(output, 0)
    save_img(step, t)

    timestep, grad, loss = guidance.train_step(text_embeddings, output)

    # writer.add_scalar("train/timestep", timestep, step)
    writer.add_scalar("train/x", norm_x, step)
    writer.add_scalar("train/y", norm_y, step)
    writer.add_scalar("train/w", norm_w, step)
    writer.add_scalar("train/h", norm_h, step)
    writer.add_histogram("grad/x", x.grad.cpu(), step)
    writer.add_histogram("grad/y", y.grad.cpu(), step)
    writer.add_histogram("grad/w", w.grad.cpu(), step)
    writer.add_histogram("grad/h", h.grad.cpu(), step)

    optimizer.step()






