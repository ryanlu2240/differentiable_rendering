# import tensorboardX
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from util import load_img_cvt_tensor, save_img, seed_everything
from models.diff_composition import DiffComposition
from loss import Customiou_loss, giou_loss

from models.sd import StableDiffusion

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    #input image argument
    parser.add_argument('--bg_img', type=str, help="bg image path")
    parser.add_argument('--src_img', type=str, help="src image path")
    #stable diffusion argument
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--prompt', type=str, help="stable diffusion prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--seed', type=int, default=42)
    #experiment workspace
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default='runs/exp1')

    opt = parser.parse_args()
    seed_everything(opt.seed)

    print(f'{opt.bg_img=}')
    print(f'{opt.src_img=}')
    print(f'{opt.prompt=}')
    
    # writer = tensorboardX.SummaryWriter(opt.workspace)
    device = 'cuda:3'
    bg_tensor = load_img_cvt_tensor(opt.bg_img).to(device)
    src_tensor = load_img_cvt_tensor(opt.src_img).to(device)

    diff_composition = DiffComposition(src_tensor.shape, bg_tensor.shape, device)
    optimizer = torch.optim.AdamW(diff_composition.parameters(), lr=1e-1)


    #load stable diffusion
    guidance = StableDiffusion(device, sd_version=opt.sd_version, fp16=False)
    guidance.register_evaluation_hooks()
    pos_embeds = guidance.get_text_embeds(opt.prompt) # [1, 77, 768]
    neg_embeds = guidance.get_text_embeds('')
    text_embeddings = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
    guidance.text_encoder.to(device)
    torch.cuda.empty_cache()

    #pseudo coor
    # gt_coor = torch.tensor([[71., 71., 431., 431.]]).to(device) # center gt
    # gt_coor = torch.tensor([[256., 0., 512., 512.]]).to(device) # right gt
    # gt_coor = torch.tensor([[0., 0., 512., 256.]]).to(device) # top gt

    base_tokens = guidance.tokenizer._tokenize(opt.prompt)
    obj_token_ids = [[torch.LongTensor([obj_token_id+1])]
                     for obj_token_id in range(len(base_tokens))]
    for step in tqdm(range(opt.iter)):
        optimizer.zero_grad()

        output, coor = diff_composition(src_tensor, bg_tensor)
        loss = guidance.train_step(text_embeddings, output, base_tokens, obj_token_ids, opt.seed, 6)
        loss.backward()

        # bounding_box_loss = Customiou_loss(coor, gt_coor) * 100 # pred, gt
        # bounding_box_loss.backward()
        
        # writer.add_scalar(f"loss/", bounding_box_loss.item(), step)

        # for name, param in diff_composition.named_parameters():
        #     if param.requires_grad:
        #         writer.add_scalar(f"train/{name}", param.data, step)
        #         writer.add_histogram(f"grad/{name}", param.grad.cpu(), step)
        #         writer.add_scalar(f"train/{name}_visualize", torch.sigmoid(param.data)*bg_tensor.shape[2], step)
        
        optimizer.step()
        save_img(step, torch.squeeze(output, 0), opt.workspace)












