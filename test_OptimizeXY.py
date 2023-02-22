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

trail_name = 'exp10'
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
# print(bg_content_tensor.shape, img_tensor.shape)

print('bg shape', bg_content_tensor.shape)
print('src shape', img_tensor.shape)

src_c, src_h, src_w = img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]
output_w, output_h = 512, 512

#construct crop source image (all zero with TIME at the center)
# src = torch.zeros((3, 964, 824))
# src[:, 452:512, 312:512] = img_tensor

src = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]+2*(output_h-img_tensor.shape[1]), img_tensor.shape[2]+2*(output_w-img_tensor.shape[2])))
src[:, output_h-img_tensor.shape[1]:output_h, output_w-img_tensor.shape[2]:output_w] = img_tensor
src = src.unsqueeze(0).to('cuda:0')



#construct mask for combine croped src and bg
# mask = torch.ones((3, 964, 824))
# mask[:, 452:512, 312:512] = 0

mask = torch.ones((img_tensor.shape[0], img_tensor.shape[1]+2*(output_h-img_tensor.shape[1]), img_tensor.shape[2]+2*(output_w-img_tensor.shape[2])))
mask[:, output_h-img_tensor.shape[1]:output_h, output_w-img_tensor.shape[2]:output_w] = 0
mask = mask.unsqueeze(0).to('cuda:0')

#construct linspace grid with size 512*512
out_y = output_w
out_x = output_h

# new_y = torch.linspace(-1, 0.06224, out_y).view(-1,1).repeat(1, out_x)
# new_x = torch.linspace(-1, 0.242718, out_x).repeat(out_y, 1)
new_y = torch.linspace(-1, -1+2/src.shape[2]*output_h, out_y).view(-1,1).repeat(1, out_x)
new_x = torch.linspace(-1, -1+2/src.shape[3]*output_w, out_x).repeat(out_y, 1)
grid = torch.cat((new_x.unsqueeze(2), new_y.unsqueeze(2)), dim=2)
grid = grid.unsqueeze(0)
grid = grid.to('cuda:0')

#example if TIME place at top left corner
# crop_src = src[:, 452:964, 312:824]
# crop_mask = mask[:, 452:964, 312:824]
# output = bg_content_tensor * crop_mask + crop_src 
# save_img(0, output) 

# xy = nn.Parameter(torch.randn(1, 2, device='cuda:0'))
# nn.init.normal_(xy)
x = nn.Parameter(torch.randn(1, device='cuda:0'))
y = nn.Parameter(torch.randn(1, device='cuda:0'))
optimizer = torch.optim.AdamW([x, y], lr=1e-1, weight_decay=0)


device = 'cuda:0'
guidance = StableDiffusion(device)

prompt = 'gaming poster'
text_embeddings = guidance.get_text_embeds(prompt, '')
guidance.text_encoder.to('cpu')
torch.cuda.empty_cache()

seed_everything(42)

num_steps = 10000
for step in tqdm(range(num_steps)):
    optimizer.zero_grad()
    # norm_xy = torch.sigmoid(xy)
    norm_x = torch.sigmoid(x)
    norm_y = torch.sigmoid(y)
    # torch.clamp(norm_xy[0][0], min=0, max=0.93775)
    # torch.clamp(norm_xy[0][1], min=0, max=0.7572)
    # norm_xy[0][0] = torch.clamp(norm_xy[0][0], min=0, max=2/src.shape[3]*(output_w-img_tensor.shape[2]))
    # norm_xy[0][1] = torch.clamp(norm_xy[0][1], min=0, max=2/src.shape[2]*(output_h-img_tensor.shape[1]))
    norm_x = norm_x.clamp(min=0, max=2/src.shape[3]*(output_w-img_tensor.shape[2]))
    norm_y = norm_y.clamp(min=0, max=2/src.shape[2]*(output_h-img_tensor.shape[1]))
    norm_xy = torch.cat((norm_x, norm_y), dim=0)
    # norm_xy = norm_xy.squeeze(0)

    writer.add_scalar("train/x", norm_x, step)
    writer.add_scalar("train/y", norm_y, step)
    # print(xy)
    # print(norm_xy)

    #apply offset on grid
    # offset_grid = grid + torch.tensor([0.7572, 0.93775]).to('cuda:0')
    offset_grid = grid + norm_xy
    
    grid_sample_src = F.grid_sample(src, grid=offset_grid, mode='bilinear')
    grid_sample_mask = F.grid_sample(mask, grid=offset_grid, mode='bilinear')

    output = bg_content_tensor * grid_sample_mask + grid_sample_src 

    t = torch.squeeze(output, 0)
    save_img(step, t)

    timestep, grad, loss = guidance.train_step(text_embeddings, output)
    writer.add_histogram("grad/x", x.grad.cpu(), step)
    writer.add_histogram("grad/y", y.grad.cpu(), step)
    # print(xy.grad)

    optimizer.step()






