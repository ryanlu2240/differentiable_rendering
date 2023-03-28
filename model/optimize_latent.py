import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL


class Optimize_Latent(nn.Module): 

    def __init__(self, device):

        super(Optimize_Latent, self).__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to(self.device)
        self.latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=self.device))



    def forward(self): 
        # print(type(self.latents))
        # print(type(self.latents * 1 / 0.18215))

        # self.latents = self.latents.data * 1 / 0.18215
        x = self.latents[:,:,:,:]
        x = 1 / 0.18215 * x
        # print('here')

        with torch.no_grad():
            imgs = self.vae.decode(x).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        # with torch.no_grad():
        # rgb = self.vae.decode_latents(self.latents)
            # img = rgb.detach().squeeze(0).permute(1,2,0).cpu().numpy()

        return imgs




