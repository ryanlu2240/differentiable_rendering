import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.functional import normalize
import math


def plot_result(idx, xgt, ygt, x1, y1):
    plt.scatter(xgt, ygt, color='red')
    plt.scatter(x1, y1, color='blue')
    plt.savefig(f'{idx}.png')


xy = nn.Parameter(torch.randn((2) , device='cuda') * 100)
print(xy.data)


optimizer = torch.optim.AdamW([xy], lr=1e-2)
loss = nn.MSELoss().to('cuda')

gt = torch.randn(2).to('cuda')
print('init')

print(f'{gt=}')
print((gt[0] ** 2 + gt[1] ** 2))
print(f'{xy.data=}')
print((xy[0] ** 2 + xy[1] ** 2))


for i in range(10000):
    optimizer.zero_grad()
    t1 = normalize(gt, p=2.0, dim = 0)
    t2 = normalize(xy, p=2.0, dim = 0)

    # print(t1,t2)


    output = loss(t1, t2)
    # print(output.item())
    output.backward()


    optimizer.step()

print('final')
print(f'{gt=}')
print((gt[0] ** 2 + gt[1] ** 2))
print(f'{xy.data=}')
print((xy[0] ** 2 + xy[1] ** 2))

print(f'gt arctan {math.atan(gt[1]/gt[0])}')
print(f'xy arctan {math.atan(xy.data[1]/xy.data[0])}')

