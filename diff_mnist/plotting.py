import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import math
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch_ddpm.ddpm.utils import repeater
from torchvision.utils import make_grid
import matplotlib


from torch_ddpm.ddpm.data.mnist import MNIST

from data import get_celeba_datasets
from data import unnormalize_fn
from torchvision.datasets import FashionMNIST


def plot_uncond_samples(img_list, uncond_samples_n_rows, uncond_samples_n_cols):
    # find dimensions of one input
    img_dims = img_list[0].size()

    # make grid of images
    # make_grid expects list of images, each of shape (C x H x W)
    # nrow is number of images per row
    grid_img = make_grid(img_list, nrow=uncond_samples_n_cols, pad_value=0,  # 0 is black padding
                                                        padding=1)  # 0 is no padding

    # convert to int for passing to imshow(...) (data received is float)
    # grid_img = grid_img.type(torch.uint8)


    if len(img_dims) == 3 and img_dims[0] == 1:
        # for each image is a 3D tensor with n_channels=1, make_grid copies the 1 channel 3 times -> just take one channel
        # see https://stackoverflow.com/questions/65616837/why-does-torchvision-utils-make-grid-return-copies-of-the-wanted-grid
        grid_img = grid_img[0, :, :]
        # afterwards, insert channel dimension again
        grid_img = grid_img.unsqueeze(0)

    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(uncond_samples_n_cols, uncond_samples_n_rows))
    ax = fig.add_subplot(1, 1, 1)
    if img_dims[0] == 1:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img, cmap='gray')  # MNIST
    else:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img)  # all other datasets
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    return fig, grid_img


def plot_downsampled_dataset(res, dataset='mnist'): 
    if dataset == 'mnist':
        ds = MNIST(load=('MNIST' in os.listdir('./datasets')), num_channels=1, device = 'cpu')
    elif dataset == 'fashion_mnist':
        ds = FashionMNIST('./datasets', train=True, download=True)
    else: 
        raise ValueError("Dataset not supported")
    ds.data = ds.data.to('cpu')

    if dataset == 'fashion_mnist': 
        # some preprocessing
        ds.data = ds.data.unsqueeze(1).float()  

    # scale original data down to the desired lower resolution res, by repeatedly applying average pooling
    for _ in range(int(math.log2(32 // res))):
        # print("downsampling")
        ds.data = torch.nn.functional.avg_pool2d(ds.data, kernel_size=(2, 2), stride=(2, 2))
    # dataloader = DataLoader(ds, batch_size=128, drop_last=True)
    # repeat_data_iter = repeater(dataloader)  # TODO why necessary? 

    # batch_x = next(repeat_data_iter)[0].to('cpu')

    # randomly permute the images
    # perm = torch.randperm(batch_x.size(0))

    N_IMAGES_PLOTTED = 16
    n_cols = int(math.sqrt(N_IMAGES_PLOTTED))
    n_rows = n_cols
    img_list = [ds.data[i].cpu().detach() for i in range(N_IMAGES_PLOTTED)]

    img_dims = img_list[0].size()

    # plotting
    grid_img = make_grid(img_list, nrow=n_cols, pad_value=1,  # padding color
                                                            padding=1)  # padding with 1 pixel

    # for each image is a 3D tensor with n_channels=1, make_grid copies the 1 channel 3 times -> just take one channel
    # see https://stackoverflow.com/questions/65616837/why-does-torchvision-utils-make-grid-return-copies-of-the-wanted-grid
    if len(img_dims) == 3 and img_dims[0] == 1:
        grid_img = grid_img[0, :, :]
        # afterwards, insert channel dimension again
        grid_img = grid_img.unsqueeze(0)

    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(n_cols, n_rows))
    ax = fig.add_subplot(1, 1, 1)
    if img_dims[0] == 1:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img, cmap='gray')  # MNIST
    else:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img)  # all other datasets
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    plt.show()
    plt.close(fig)

    return grid_img



def plot_downsampled_dataset_celeba(res, ds, norm_mean, norm_std): 
    ds.data = ds.data.to('cpu')
    # scale original data down to the desired lower resolution res, by repeatedly applying average pooling
    for _ in range(int(math.log2(64 / res))):
        # print("downsampling")
        ds.data = torch.nn.functional.avg_pool2d(ds.data, kernel_size=(2, 2), stride=(2, 2))
    dataloader = DataLoader(ds, batch_size=128, drop_last=True)
    repeat_data_iter = repeater(dataloader)  # TODO why necessary? 

    batch_x = next(repeat_data_iter).to('cpu')

    # unnormalize data
    # batch_x = unnormalize_fn(batch_x, norm_mean, norm_std)

    # convert to int
    batch_x = batch_x.int()

    # randomly permute the images
    # perm = torch.randperm(batch_x.size(0))

    N_IMAGES_PLOTTED = 16
    n_cols = int(math.sqrt(N_IMAGES_PLOTTED))
    n_rows = n_cols
    img_list = [batch_x[i].cpu().detach() for i in range(N_IMAGES_PLOTTED)]

    img_dims = img_list[0].size()

    # plotting
    grid_img = make_grid(img_list, nrow=n_cols, pad_value=1,  # padding color
                                                            padding=1)  # padding with 1 pixel

    # for each image is a 3D tensor with n_channels=1, make_grid copies the 1 channel 3 times -> just take one channel
    # see https://stackoverflow.com/questions/65616837/why-does-torchvision-utils-make-grid-return-copies-of-the-wanted-grid
    if len(img_dims) == 3 and img_dims[0] == 1:
        grid_img = grid_img[0, :, :]
        # afterwards, insert channel dimension again
        grid_img = grid_img.unsqueeze(0)

    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(n_cols, n_rows))
    ax = fig.add_subplot(1, 1, 1)
    if img_dims[0] == 1:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img, cmap='gray')  # MNIST
    else:
        # imshow(...) expects (H,W) or (H,W,3)
        ax.imshow(grid_img)  # all other datasets
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    plt.show()
    plt.close(fig)

    return grid_img



def plot_unet_norms(norms_list, t_list): 
    """
    Similar plot: anonymised
    """
 
    # aggregate all batches into a single dictionary
    n_levels = len(norms_list[0]['down'])
    # n_norms = 0
    norms_agg = {
                'down': {k: [] for k in range(n_levels)},
                'middle': [],
                'up': {k: [] for k in range(n_levels)},
            }
    for i, norms in enumerate(norms_list): 
        # down
        for level, n_list in sorted(list(norms['down'].items()), key=lambda x: x[0]):
            for j, n in enumerate(n_list): 
                if i == 0: 
                    norms_agg['down'][level].append(n)
                else: 
                    norms_agg['down'][level][j] = torch.cat((norms_agg['down'][level][j], n), dim=0)
                # n_norms += 1
        # middle
        for j, n in enumerate(norms['middle']):
            if i == 0: 
                norms_agg['middle'].append(n)
            else: 
                norms_agg['middle'][j] = torch.cat((norms_agg['middle'][j], n), dim=0)
            # n_norms += 1
        # up
        for level, n_list in sorted(list(norms['up'].items()), key=lambda x: x[0], reverse=True):
            for j, n in enumerate(n_list): 
                if i == 0: 
                    norms_agg['up'][level].append(n)
                else: 
                    norms_agg['up'][level][j] = torch.cat((norms_agg['up'][level][j], n), dim=0)
                # n_norms += 1

    # prepare labels for the plot
    interval_labels = []
    n_enc_blocks, n_mid_blocks, n_dec_blocks, cum_n_blocks = 0, 0, 0, 0
    # level_to_res = {0: "32x32", 1: "16x16", 2: "8x8", 3: "4x4"}

    # aggregate all norms, across 'down'/'middle'/'up' and all levels, into a single tensor
    highest_level = max(norms_agg['down'].keys())
    batch_dim = norms_agg['down'][highest_level][0].size()[0]
    device = norms_agg['down'][highest_level][0].device
    norms_tensor = torch.empty((batch_dim, 0)).to(device)   # (batch_dim, n_norms)
    # down
    for level, n_list in sorted(list(norms_agg['down'].items()), key=lambda x: x[0]):  # forward in levels
        n_res_blocks_level = 0
        for n in norms_agg['down'][level]:
            norms_tensor = torch.cat((norms_tensor, n.unsqueeze(1)), dim=1)
            n_res_blocks_level += 1
        cum_n_blocks += n_res_blocks_level
        interval_labels.append(["level " + str(level) + " (enc)", cum_n_blocks])
        n_enc_blocks += n_res_blocks_level
    # middle
    for n in norms_agg['middle']:
        norms_tensor = torch.cat((norms_tensor, n.unsqueeze(1)), dim=1)
        n_mid_blocks += 1
    cum_n_blocks += n_mid_blocks
    interval_labels.append(("level " + str(highest_level) + " (mid)", cum_n_blocks))
    # up
    for level, n_list in sorted(list(norms_agg['up'].items()), key=lambda x: x[0], reverse=True):  # reverse in levels
        n_res_blocks_level = 0
        for n in norms_agg['up'][level]:
            norms_tensor = torch.cat((norms_tensor, n.unsqueeze(1)), dim=1)
            n_res_blocks_level += 1
        cum_n_blocks += n_res_blocks_level
        interval_labels.append(["level " + str(level) + " (dec)", cum_n_blocks])
        n_dec_blocks += n_res_blocks_level
    n_norms = norms_tensor.size()[1]

    # aggregate time steps
    t_agg = t_list[0]
    for i, t in enumerate(t_list[1:]):
        t_agg = torch.cat((t_agg, t), dim=0)
    
    # normalize t to be in [0,1] (if required)
    if t_agg.max() > 1: 
        t_agg = t_agg / t_agg.max()

    # 'bin' the timesteps over a grid
    n_bins = 5
    t_bins = torch.linspace(0, t_agg.max(), n_bins).to(device)  # shape: (n_bins + 1,), hence +1 not necessary (already done)
    t_bin_idx = torch.bucketize(input=t_agg, boundaries=t_bins)

    # compute average and standard deviation for norms in each bin
    bin_idx_to_norm_avg_tensor = {}
    for bin_idx in range(n_bins):
        bin_idx_to_norm_avg_tensor[bin_idx] = norms_tensor[t_bin_idx == bin_idx].mean(dim=0).detach().cpu().numpy()  
    bin_idx_to_norm_std_tensor = {}
    for bin_idx in range(n_bins):
        bin_idx_to_norm_std_tensor[bin_idx] = norms_tensor[t_bin_idx == bin_idx].std(dim=0).detach().cpu().numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(8, 6))  
    ax = fig.add_subplot(1, 1, 1)

    # colour scale
    cmap = plt.cm.cividis  # sequential colour map since time has an ordering  ; YlOrBr
    norm = matplotlib.colors.Normalize(vmin=t_bins.min(), vmax=t_bins.max())
    
    for bin_idx, bin_idx_to_norms_avg_tensor in bin_idx_to_norm_avg_tensor.items():
        # format the following plot command in the  t = to have 2 decimal places
        ax.plot(np.arange(n_norms), bin_idx_to_norms_avg_tensor, label='t = %.2f' % t_bins[bin_idx], color = cmap(norm(t_bins[bin_idx].detach().cpu().numpy())))   # TODO check in index off by one? 

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # y as logscale
    ax.set_yscale('log')

    plt.xlabel("ResNet layer index")
    plt.ylabel("L^2 norm of ResNet state (normalised by dimension)")

    # vertical lines and labeling of the resolutions/levels
    bottom_ylim, top_ylim = ax.get_ylim()
    prev_lower_bound = 0
    for label in interval_labels:
        ax.axvline(x=label[1] - 0.5, color='black', linestyle='--', linewidth=0.5)
        # x position: (label[1] + prev_lower_bound)/2.
        ax.text(label[1] - 0.5 - 0.5, bottom_ylim + top_ylim * 0.01, label[0], horizontalalignment='center', verticalalignment='center', fontsize=8, rotation=90)  # , rotation=90, verticalalignment='bottom'
        
        prev_lower_bound = label[1]
        # ax.text(label[1], 0.1, label[0], rotation=90, verticalalignment='bottom')


    # dashed horizontal line at 0
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])  # left out because of legend above

    # plt.show()


    return fig 

    # TODO interval plots might be "one off" (check where up and down sample operations are)
    # TODO normalize plot
    # TODO separate encoder and decoder
    # TODO fix t = intervals
    # ----
    # TODO adjust size of plot depending on size of network
    # TODO hwo to catch if no elements in bin?
    # TODO check if boundaries are correct
    # TODO plot per time interval, and also global average
    # TODO STD: how many STD to plot? --> as in HVAE plot
    # TODO look through HVAE plot in general: horizontal lines etc. 


    # # indices
    # norms_indices = {
    #     'down': {k: [] for k in range(n_levels)},
    #     'middle': [],
    #     'up': {k: [] for k in range(n_levels)},
    # }
    # index = 0
    # # down
    # for level in range(n_levels):
    #     norms_indices['down'][level] = list(range(index, index + len(norms_agg['down'][level])))
    #     index += len(norms_agg['down'][level])
    # # middle
    # norms_indices['middle'] = list(range(index, index + len(norms_agg['middle'])))
    # index += len(norms_agg['middle'])
    # # up
    # for level in reversed(list(range(n_levels))):  # reverse list, starting with lowest levels first
    #     norms_indices['up'][level] = list(range(index, index + len(norms_agg['up'][level])))
    #     index += len(norms_agg['up'][level])


    

    # bin idx to norm object
    # bin_idx_to_right_bound_t = {i: t_bins[i] for i in range(len(t_bins))}  # TODO is this the right bound?
    # bin_idx_to_norms = {i: None for i in range(len(t_bins))}
    # for bin_idx, right_bound_t in enumerate(t_bins[1:].tolist()): 
    #     # initialise 
    #     bin_idx_to_norms[bin_idx] = {
    #             'down': {k: [] for k in range(n_levels)},
    #             'middle': [],
    #             'up': {k: [] for k in range(n_levels)},
    #         }
    #     # down 
    #     for level, n_list in norms['down'].items():
    #         for j, n in enumerate(n_list): 
    #             bin_idx_to_norms[bin_idx]['down'][level].append(n[t_agg == bin_idx])
    #     # middle
    #     for j, n in enumerate(norms['middle']):
    #         bin_idx_to_norms[bin_idx]['middle'].append(n[t_agg == bin_idx])
    #     # up
    #     for level, n_list in norms['up'].items():
    #         for j, n in enumerate(n_list): 
    #             bin_idx_to_norms[bin_idx]['up'][level].append(n[t_agg == bin_idx])
    
    # # average for each bin obj
    # bin_idx_to_norms_avg = {i: None for i in range(len(t_bins))}
    # for bin_idx, right_bound_t in t_bins[1:]: 
    #     # initialise 
    #     bin_idx_to_norms[bin_idx] = {
    #             'down': {k: [] for k in range(n_levels)},
    #             'middle': [],
    #             'up': {k: [] for k in range(n_levels)},
    #         }
    #     # down
    #     for level, n_list in norms['down'].items():
    #         for j, n in enumerate(n_list): 
    #             bin_idx_to_norms[bin_idx]['down'][level].append(torch.mean(n))
    #     # middle
    #     for j, n in enumerate(norms['middle']):
    #         bin_idx_to_norms[bin_idx]['middle'].append(torch.mean(n))
    #     # up
    #     for level, n_list in norms['up'].items():
    #         for j, n in enumerate(n_list): 
    #             bin_idx_to_norms[bin_idx]['up'][level].append(torch.mean(n))
        
    # # list to tensor
    # for bin_idx in bin_idx_to_norms_avg.keys(): 
    #     # down
    #     for level, n_list in bin_idx_to_norms_avg[bin_idx]['down'].items():
    #         bin_idx_to_norms_avg[bin_idx]['down'][level] = torch.stack(n_list)
    #     # middle
    #     bin_idx_to_norms_avg[bin_idx]['middle'] = torch.stack(bin_idx_to_norms_avg[bin_idx]['middle'])
    #     # up
    #     for level, n_list in bin_idx_to_norms_avg[bin_idx]['up'].items():
    #         bin_idx_to_norms_avg[bin_idx]['up'][level] = torch.stack(n_list)

    # # one tensor per bin
    # bin_idx_to_norms_avg_tensor = {i: None for i in range(len(t_bins))}
    # for bin_idx in bin_idx_to_norms_avg_tensor.keys():
    #     # down
    #     downs = torch.cat([bin_idx_to_norms_avg[bin_idx]['down'][level] for level in range(n_levels)], dim=0)
    #     # middle
    #     middle = bin_idx_to_norms_avg[bin_idx]['middle']
    #     # up
    #     ups = torch.cat([bin_idx_to_norms_avg[bin_idx]['up'][level] for level in reversed(list(range(n_levels)))], dim=0)  # reverse list 
    #     # concat
    #     bin_idx_to_norms_avg_tensor[bin_idx] = torch.cat((downs, middle, ups), dim=0)
    
    

    # ----------------------------------------------- OLD -----------------------------------------------



    # print("l2 norm of output of bottom-up block, block idx %d, res %d: "%(i, x.shape[2]), torch.mean(torch.linalg.norm(torch.flatten(x, start_dim=1), dim=1)).item())
    # norm_batch = torch.linalg.norm(torch.flatten(x, start_dim=1), dim=1).cpu()
    # norm_batch_dict[i].append(norm_batch)
    # # calculating some stuff for plotting
    # condition = True
    # n_batches = 10   # to be chosen by user
    # reps_per_res = compute_blocks_per_res(self.H.enc_blocks)
    # n_blocks = 0
    # for reps in reps_per_res.values():
    #     n_blocks += reps
    # for idx in range(n_blocks):   # TODO number of blocks
    #     condition = condition and len(norm_batch_dict[idx]) >= n_batches
    # if condition:
    #     print("plotting the norms")
    #     # take first n_batches
    #     for idx in range(max_n_idxs):
    #         norm_batch_dict[idx] = norm_batch_dict[idx][:n_batches]
    #     # some placeholders
    #     norm_mean = []
    #     norm_std = []
    #     largest_idx = -1
    #     for idx in range(max_n_idxs):
    #         if len(norm_batch_dict[idx]) > 0:
    #             if idx > largest_idx:
    #                 largest_idx = idx
    #             # concatenate all the data points across batches
    #             norm_batch_dict[idx] = torch.cat(norm_batch_dict[idx])
    #             # compute mean and std for each block
    #             norm_mean.append(torch.mean(norm_batch_dict[idx]).item())
    #             norm_std.append(torch.std(norm_batch_dict[idx]).item())
    #     norm_mean = np.array(norm_mean)
    #     norm_std = np.array(norm_std)
    #     lower = norm_mean - 2 * norm_std
    #     upper = norm_mean + 2 * norm_std
    #     layer_idxs = np.arange(n_blocks)
    #     # do the plotting
    #     plt.plot(layer_idxs, norm_mean, color='black', linewidth=2)
    #     plt.plot(layer_idxs, lower, color='black', linewidth=.5)
    #     plt.plot(layer_idxs, upper, color='black', linewidth=.5)
    #     # horizontal lines indicating the resolution jumps
    #     res_to_reps_tuples = [(key, val) for key, val in reps_per_res.items()]
    #     res_to_reps_tuples = sorted(res_to_reps_tuples, key=lambda tup: tup[0], reverse=True)
    #     reps = [tup[1] for tup in res_to_reps_tuples]
    #     resolutions = [tup[0] for tup in res_to_reps_tuples]
    #     jumps = (np.cumsum(reps) - 0.5).tolist()[:-1]
    #     for jump in jumps:
    #         plt.axvline(x=jump, color='g', linestyle='--')
    #     plt.xlabel("Bottom-up block $i$")
    #     plt.ylabel("$\|\| x_i \|\|_2$ with output $x_i$ of bottom-up block")
    #     plt.xlim(0, np.max(layer_idxs))
    #     plt.ylim(0, np.max(upper))
    #
    #     y_text_pos = 0.9 * np.max(upper)
    #     cum_x_pos = 0
    #     for i, jump in enumerate([0] + jumps):
    #         label = str(resolutions[i]) + 'x' + str(resolutions[i])
    #         if i == 0:
    #             shift = 0.5
    #         else:
    #             shift = 0.5
    #         plt.text(jump + shift, y_text_pos, label, fontsize=6, color='green', rotation='vertical')
    #     plt.savefig("/home/user/VDVAE/plotting/cifar_state_norm_forward.pdf", dpi=600)
    #     plt.show()
    #     print("done with plotting")