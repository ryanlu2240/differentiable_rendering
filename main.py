import tensorboardX
import torch
import argparse
import numpy as np
from tqdm import tqdm
import json

from torchvision import transforms
from util import load_img_cvt_tensor, save_img, seed_everything, load_mulimg_cvt_tensor
from model.diff_composition import DiffComposition
from model.optimize_latent import Optimize_Latent
from loss import Customiou_loss, giou_loss

from sd import StableDiffusion

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    #input image argument
    parser.add_argument('--bg_img', type=str, help="bg image path")
    parser.add_argument('--src_img', type=str, help="src image path")
    parser.add_argument('--src_imgs_folder', type=str, help="src images folder path")
    #stable diffusion argument
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--prompt', type=str, help="stable diffusion prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    #training argument
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    #experiment workspace
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default='runs/exp1')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    print(vars(opt))

    seed_everything(opt.seed)
    
    writer = tensorboardX.SummaryWriter(opt.workspace)
    with open(f'{opt.workspace}/arguments.json', 'w') as f:
        json.dump(vars(opt), f, indent=4)

    device = 'cuda:0'
    bg_tensor = load_img_cvt_tensor(opt.bg_img).to(device)
    # src_tensor = load_img_cvt_tensor(opt.src_img).to(device)
    src_tensor = load_mulimg_cvt_tensor(opt.src_imgs_folder).to(device)

    
    


    #load stable diffusion
    guidance = StableDiffusion(device, opt.sd_version)
    text_embeddings = guidance.get_text_embeds(opt.prompt, opt.negative)
    guidance.text_encoder.to(device)
    torch.cuda.empty_cache()

    #init differenable composition
    diff_composition = DiffComposition(src_tensor.shape, bg_tensor.shape, device)
    #init Optimize_Latent
    # optimize_latent = Optimize_Latent(device='cuda:1')

    

    #pseudo coor
    # gt_coor = torch.tensor([[71., 71., 431., 431.]]).to(device) # center gt
    # gt_coor = torch.tensor([[256., 0., 512., 512.]]).to(device) # right gt
    # gt_coor = torch.tensor([[0., 0., 512., 256.]]).to(device) # top gt


    # optimizer = torch.optim.AdamW(list(optimize_latent.parameters()) + list(diff_composition.parameters()), lr=1e-1)
    # optimizer = torch.optim.AdamW(optimize_latent.parameters(), lr=1e-1)
    optimizer = torch.optim.AdamW(diff_composition.parameters(), lr=opt.lr)

    for step in tqdm(range(opt.iter)):
        optimizer.zero_grad()

        output = diff_composition(src_tensor, bg_tensor)
        # output = optimize_latent()
        loss = guidance.train_step(text_embeddings, output)

        # iou loss
        # bounding_box_loss = Customiou_loss(coor, gt_coor) * 100 # pred, gt
        # bounding_box_loss.backward()
        # writer.add_scalar(f"loss/", bounding_box_loss.item(), step)
        # print(optimize_latent.latents.grad)
        # writer.add_histogram(f"grad/coor", param.grad.cpu(), step)

        # for name, param in diff_composition.named_parameters():
        #     if param.requires_grad:
        #         writer.add_scalar(f"train/{name}", param.data, step)
        #         writer.add_histogram(f"grad/{name}", param.grad.cpu(), step)
        #         writer.add_scalar(f"train/{name}_visualize", torch.sigmoid(param.data)*bg_tensor.shape[2], step)
        
        optimizer.step()
        save_img(step, torch.squeeze(output, 0), opt.workspace)














