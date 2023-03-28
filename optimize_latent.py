import math
from tqdm import tqdm
import torch
import torch.nn as nn
from sd import StableDiffusion
from util import seed_everything
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt

seed_everything(42)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


device = 'cuda:0'
guidance = StableDiffusion(device)
# guidance.vae.encoder = None

prompt = 'a white background with a newspaper title at the bottom of the canvas'
text_embeddings = guidance.get_text_embeds(prompt, '')
guidance.text_encoder.to('cpu')
torch.cuda.empty_cache()

seed_everything(42)
latents = nn.Parameter(torch.randn(1, 4, 64, 64, device='cuda:2'))
optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
num_steps = 1000
scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps*1.5))

for step in tqdm(range(num_steps)):
    optimizer.zero_grad()

    imgs = guidance.decode_latents(latents)
    loss = guidance.train_step(text_embeddings, imgs)
    print(latents.grad)

    # t = torch.randint(guidance.min_step, guidance.max_step + 1, [1], dtype=torch.long, device='cuda:2')
    # with torch.no_grad():
    #     # add noise
    #     noise = torch.randn_like(latents).to("cuda:2")
    #     latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
    #     # pred noise
    #     latent_model_input = torch.cat([latents_noisy] * 2)
    #     noise_pred = guidance.unet(latent_model_input.to("cuda:2"), t, encoder_hidden_states=text_embeddings.to("cuda:2")).sample

    # # perform guidance (high scale from paper!)
    # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    # noise_pred = noise_pred_uncond + 100 * (noise_pred_text - noise_pred_uncond)

    # w = (1 - guidance.alphas[t]).to("cuda:2")
    # grad = w * (noise_pred - noise)

    # latents.backward(gradient=grad, retain_graph=True)

    optimizer.step()
    scheduler.step()

    if step > 0 and step % 100 == 0:
        rgb = guidance.decode_latents(latents)
        img = rgb.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        print('[INFO] save image', img.shape, img.min(), img.max())
        plt.imsave(f'lat_img_{step}.jpg', img)