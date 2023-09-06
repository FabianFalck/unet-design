from tqdm import tqdm
from torch_ddpm.ddpm.utils import repeater
import torch


def train(model, diffusion, optimizer, dataloader, num_iterations, device):
    repeat_data_iter = repeater(dataloader)

    for it in tqdm(range(num_iterations)):   # progress bar
        batch_x = next(repeat_data_iter)[0].to(device)

        # Version 1 - regular
        t = diffusion.sample_t(batch_x)  # samples time indices
        # Version 2 - hijacked
        # t_part1 = diffusion.sample_t(batch_x)
        # t_part2 = torch.tensor(diffusion.N-1).repeat(t_part1.shape[0])
        # t = torch.cat((t_part1[:int(t_part1.shape[0]/2)], t_part2[:int(t_part1.shape[0]/2)]), dim=0)
        # print(t.shape)

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
