import os, sys
sys.path.append('.')
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from ddpm.data.mnist import MNIST
from ddpm.models.utils import get_mlpnet, get_unet
from ddpm.train import train
from ddpm.diffusion import Diffusion

from tqdm import tqdm
from torch_ddpm.ddpm.utils import repeater
import torch

from utils import load_dict_from_yaml


import wandb


# load wandb api key, project name and entity name
wandb_config = load_dict_from_yaml('wandb.yml')
# login to wandb and create run
wandb.login(key=wandb_config['user'])
wandb_run = wandb.init(project=wandb_config['project_name'], entity=wandb_config['team_name'], mode='online')



# config
learning_rate = 1e-3
batch_size = 128
num_iterations = 10**4
N =30
device='cuda'

# init objects
diffusion = Diffusion(N=30).to(device)
model  = get_unet(28,1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

ds = MNIST(load=('MNIST' in os.listdir('./datasets')), num_channels=1, device = device)
ds.data = ds.data.to(device)
dataloader = DataLoader(ds, batch_size=batch_size, drop_last=True)

# train
repeat_data_iter = repeater(dataloader)

for it in tqdm(range(num_iterations)):   # progress bar
    batch_x = next(repeat_data_iter)[0].to(device)

    # Version 1 - regular
    t = diffusion.sample_t(batch_x)  # samples time indices
    # Version 2 - hijacked
    # t_part1 = diffusion.sample_t(batch_x)
    # t_part2 = torch.tensor(diffusion.N-1).repeat(t_part1.shape[0]).to(device)
    # t = torch.cat((t_part1[:int(t_part1.shape[0]/2)], t_part2[:int(t_part1.shape[0]/2)]), dim=0)

    # forward of data by t timesteps --> produces x_t
    perturbed_sample = diffusion.sample_x(batch_x, t)
    x_t = perturbed_sample.x_t
    # print("iter", it)
    # print(x_t)
    model_out = model(x_t, t.unsqueeze(-1))  # model receives x_t and t as input

    optimizer.zero_grad()

    # noise prediction
    # perturbed_sample.z == noise
    # model_out is the noise prediction of the model
    loss = diffusion.loss(model_out, perturbed_sample.z)

    loss.backward()

    optimizer.step()

    if it % 500 == 0:
        # plot
        x_T = torch.randn((16,1,28,28), device=device)
        x_0, x_mean = diffusion.reverse_sample(x_T, model)
        plot_x = x_mean.cpu().detach().numpy()

        for i in range(16):
            x = plot_x[i].reshape(28,28)
            plt.clf()
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(x, 'gray')
            wandb.log({'samples': wandb.Image(plt)}, step=it)
            # print("logged")
            plt.close(fig)
            # plt.show()

