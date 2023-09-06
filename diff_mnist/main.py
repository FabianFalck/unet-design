import os, sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
import math

from torch_ddpm.ddpm.data.mnist import MNIST
from torch_ddpm.ddpm.diffusion import Diffusion
from torch_ddpm.ddpm.utils import repeater
from torch_ddpm.ddpm.models.utils import get_mlpnet, get_unet
from pytorch_wavelets import DWTForward, DWTInverse

from mnist_diff.models import get_unet_wavelet_enc

from utils import load_dict_from_yaml, download_some_wandb_files

from hyperparams import args_parser, check_hyperparams
from plotting import plot_uncond_samples
from mnist_diff.layers import ScoreNetwork, UpInterpolate
from data import get_celeba_datasets, unnormalize_fn

from mnist_diff.unet_wavelet import UNet_wavelet, get_unet_wavelet
from plotting import plot_unet_norms

from data import MNIST_Triangular
from data import Preprocess_triangular, swap_array


def main(): 
    H = args_parser()
    # check hyperparams for consistency etc.
    check_hyperparams(H)

    # make entire code deterministic
    np.random.seed(H.SEED)
    torch.manual_seed(H.SEED)
    torch.cuda.manual_seed(H.SEED)

    # load wandb api key, project name and entity name
    wandb_config = load_dict_from_yaml('setup/wandb.yml')
    # login to wandb and create run
    wandb.login(key=wandb_config[H.USER])
    wandb_run = wandb.init(project=wandb_config['project_name'], entity=wandb_config['team_name'], mode=H.WANDB_MODE)

    # auxiliary / remembering variables
    RESTORE = H.TRAIN_ID is not None or H.TEST_ID is not None
    training = H.TEST_ID is None
    train_iter = H.train_iter
    test_iter = H.test_iter

    # download some files first and load them
    if RESTORE:
        files_to_restore = ["H.dict", "last_save_iter.th"]  # 'last_save_iter.th' is just storing an int
        run_id = H.TRAIN_ID if H.TRAIN_ID is not None else H.TEST_ID
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=run_id)
        # Note: loads another H dictionary in the case of restoring which overwrites the new one above
        H = torch.load(os.path.join(wandb.run.dir, files_to_restore[0]))  # overwrites H parsed above
        last_save_iter = torch.load(os.path.join(wandb.run.dir, files_to_restore[1]))
        # In the restored H, we overwrite train or test restore information which we need below
        if training:
            H.TRAIN_ID = run_id
            H.train_iter = train_iter
        else:
            H.TEST_ID = run_id
            H.test_iter = test_iter
        print("Note: Restoring run " + run_id + ". Any passed command line arguments are ignored!")   # Note: Could even throw exception if this is the case.

    if H.TRAIN_ID is not None:
        train_iter = last_save_iter if H.train_iter is None else H.train_iter
        H.restore_iter = train_iter
        model_load_file_name = 'iter-%d-model.th'%train_iter
        # model_eval_load_file_name = 'iter-%d-model_eval.th'%train_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%train_iter
        # scheduler_load_file_name = 'iter-%d-scheduler.th'%train_iter
        files_to_restore = [model_load_file_name, optimizer_load_file_name]  #  model_eval_load_file_name, scheduler_load_file_name
        download_run_id = H.TRAIN_ID
    elif H.TEST_ID is not None:
        test_iter = last_save_iter if H.test_iter is None else H.test_iter
        H.restore_iter = test_iter
        # Note: could only load model_eval here
        model_load_file_name = 'iter-%d-model.th'%test_iter
        # model_eval_load_file_name = 'iter-%d-model_eval.th'%test_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%test_iter
        # scheduler_load_file_name = 'iter-%d-scheduler.th'%test_iter
        files_to_restore = [model_load_file_name, optimizer_load_file_name]  # model_eval_load_file_name, scheduler_load_file_name
        download_run_id = H.TEST_ID
    else:
        train_iter = 0

    if RESTORE:
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=download_run_id)


    # make device a global variable so that dataset.py can access it
    global device
    # initializing global variable (see above)
    device = torch.device(H.DEVICE)

    # save the hyperparam config after everything was set up completely (some values of H will only be written at run-time, but everything is written at this point)
    if not H.test_eval and not H.train_restore:
        print("saving H config...")
        torch.save(H, os.path.join(wandb.run.dir, "H.dict"))

    
    # for legacy reasons this assigned
    res, beta_min, beta_max, N, eps, T = H.RESOLUTION, H.BETA_MIN, H.BETA_MAX, H.N, H.EPS, H.T

    print("Data resolution {}.".format(res))

    # dataset specific configs
    if H.DATASET == 'mnist':
        in_channels =  1
        dataset_res = 32
    elif H.DATASET == 'mnist_triangular':
        in_channels = 1
        dataset_res = 64
    elif H.DATASET == 'celeba':
        in_channels = 3
        dataset_res = 64

    # initialize score model
    # U-Net
    # if H.MODEL == 'unet': 
    #     model  = get_unet(H.RESOLUTION, in_channels, num_channels=H.NUM_CHANNELS, dropout=H.DROPOUT, num_res_blocks=H.NUM_RES_BLOCKS).to(H.DEVICE)
    # MLP
    if H.MODEL == 'mlp':
        in_dim = res * res * 1  # 1 channel   # TODO make input dependent hyperparam
        kwargs = {
            "encoder_layers": H.ENCODER_LAYERS,
            "pos_dim": H.POS_DIM,
            "decoder_layers": H.DECODER_LAYERS,
            "x_dim": in_dim,
        }
        model = ScoreNetwork(**kwargs).to(device)  # TODO dimensions / input correct? 
    # elif H.MODEL == 'unet_wavelet_enc': 
    #     model  = get_unet_wavelet_enc(H.RESOLUTION, in_channels, num_channels=H.NUM_CHANNELS, dropout=H.DROPOUT, num_res_blocks=H.NUM_RES_BLOCKS).to(H.DEVICE)
    # elif H.MODEL == 'unet_wavelet': 
    elif H.MODEL == 'unet_wavelet':
        model = get_unet_wavelet(H.RESOLUTION, in_channels, num_channels=H.NUM_CHANNELS, dropout=H.DROPOUT, num_res_blocks=H.NUM_RES_BLOCKS, dwt_encoder=H.DWT_ENCODER, multi_res_loss=H.MULTI_RES_LOSS, model_out_passed_on=H.MODEL_OUT_PASSED_ON, avg_pool_down=H.AVG_POOL_DOWN).to(H.DEVICE)

    else: 
        raise Exception("Chosen model not implemented.")
    # restore model state
    if RESTORE:
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, model_load_file_name)))

    # initialize diffusion
    diffusion = Diffusion(beta_min=beta_min, beta_max=beta_max, N=N, eps=eps, T=T, multi_res_loss=H.MULTI_RES_LOSS, weighted_multi_res_loss=H.WEIGHTED_MULTI_RES_LOSS).to(H.DEVICE)
    

    preprocessor = Preprocess_triangular(J=int(math.log2(dataset_res)))
    
    # initialize dataset and loader
    if H.DATASET == 'mnist':
        ds = MNIST(load=('MNIST' in os.listdir('./datasets')), num_channels=1, device = H.DEVICE)
    elif H.DATASET == 'mnist_triangular':
        # TODO possibly normalise? 
        ds = MNIST_Triangular(root = './data', train=True, download=True, to_square_preprocess=H.to_square_preprocess)
    elif H.DATASET == 'celeba':
        dataset_train, dataset_val, dataset_test, norm_mean, norm_std = get_celeba_datasets()
        # TODO for now, just use the train set
        ds = dataset_train

    # data to GPU
    # TODO for now, just training data to GPU
    ds.data = ds.data.to(H.DEVICE)

    # scale original data down to the desired lower resolution res, by repeatedly applying average pooling
    for _ in range(int(math.log2(dataset_res // res))):
        print("downsampling")
        ds.data = torch.nn.functional.avg_pool2d(ds.data, kernel_size=(2, 2), stride=(2, 2))
    dataloader = DataLoader(ds, batch_size=H.BATCH_SIZE, drop_last=True)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=H.LR)
    # optimizers must be constructed after the model is moved to GPU, see https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/3?u=shaibagon
    if RESTORE:
        optimizer.load_state_dict(torch.load(os.path.join(wandb.run.dir, optimizer_load_file_name)))

    # weights&biases tracking (gradients, network topology)
    # only watch model (this is the one doing the training step)
    # only do so when training
    if H.TEST_ID is None:
        wandb.watch(model)

    # count total number of parameters and print
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)  #  ; 7754180
    params_string = f'{params_count:,}'
    print("Number of parameters of model: " + params_string)
    # store number of parameters count in configs
    H.n_params = params_string

    if len(H.NUM_ITERATIONS_LIST) > 1: 
        SEQU_TRAIN_ALGO = True
        assert model.n_levels == len(H.NUM_ITERATIONS_LIST)
        print("In multi-resolution mode!!!")
    else: 
        SEQU_TRAIN_ALGO = False

    # print H and run dir once everything is set up and completed
    print(H)
    print("wandb.run.dir: ", wandb.run.dir)
    # update configs -> remember hyperparams; only when not testing
    if H.TEST_ID is None:
        wandb.config.update(H)


    # print("Number of resolutions during learning: ", len(H.NUM_ITERATIONS))  # cascaded

    repeat_data_iter = repeater(dataloader)  # TODO why necessary? 

    # For testing purposes only
    last_norm_out = torch.tensor(-1.)
    x_T_4_test = torch.randn((25, 1, 4, 4), device=H.DEVICE)

    # DO HERE MULTI-RES LOOP
    for j, num_iters in enumerate(H.NUM_ITERATIONS_LIST): 

        # slightly weird condition
        if H.MODEL in ['unet_wavelet', 'unet'] and (SEQU_TRAIN_ALGO or H.MULTI_RES_LOSS): 
            # some variables for sampling
            highest_res = H.RESOLUTION
            resolutions = [highest_res // (2 ** i) for i in range(model.n_levels)]
            all_possible_unet_resolutions = resolutions.copy()
            res_to_n_levels_used = {res: model.n_levels - int(math.log2(dataset_res/res)) for res in resolutions}
            cur_res = resolutions[-(j+1)]
            if SEQU_TRAIN_ALGO: 
                # only consider the highest resolutions at this stage
                resolutions = resolutions[(-j)-1:]
        else: 
            resolutions = [H.RESOLUTION]
            cur_res = H.RESOLUTION
            res_to_n_levels_used = {H.RESOLUTION: model.n_levels}
        

        if SEQU_TRAIN_ALGO:
            n_levels_used_training = j + 1
        else: 
            n_levels_used_training = model.n_levels

        print("--- cur_res = {} (j = {})...".format(cur_res, j))

        # freeze parameters of all except the finest level used in decoder in this stage of the sequential training algorithm
        if H.FREEZE_LOWER_RES: 
            # TESITNG PURPOSES: Freeze all parameters
            # for p, param in enumerate(model.parameters()):
            #     if p != 0:   # must leave one param active
            #         param.requires_grad = False
            #         param.grad = None


            assert SEQU_TRAIN_ALGO
            # decoder blocks
            for level in list(range(len(model.channel_mult)))[::-1][:n_levels_used_training - 1]: 
                for out_block in model.out_f_list[level]: 
                    print("j {}, freeze Dec # : level {}. out_blo {}".format(j, level, out_block[0].__class__.__name__))  # taking 0-th layer of Sequential object
                    for param in out_block.parameters():
                        param.grad = None
                        param.requires_grad = False
                        
            for level in list(range(len(model.channel_mult)))[::-1][:max(n_levels_used_training - 2, 0)]:  # the upsample layer on the coarsest level is not frozen, hence different range
                print("j {}, freeze Dec # : level {}. out_ups {}".format(j, level, model.out_upsample_list[level][0].__class__.__name__))
                for param in model.out_upsample_list[level][0].parameters():
                    param.grad = None
                    param.requires_grad = False
            for p in range(n_levels_used_training - 1): 
                print("j {}, freeze Dec # : p {}. out_act {}".format(j, p, model.out_activation_list[p].__class__.__name__))
                for param in model.out_activation_list[p].parameters(): 
                    param.grad = None
                    param.requires_grad = False
                print("j {}, freeze Dec # : p {}. out_rch {}".format(j, p, model.out_reduce_channels_list[p].__class__.__name__))
                for param in model.out_reduce_channels_list[p].parameters(): 
                    param.grad = None
                    param.requires_grad = False
            # middleblocks (part of decoder)
            if n_levels_used_training >= 2: 
                print("j {}, freeze Mid % : middle_block {}".format(j, model.middle_block.__class__.__name__))
                for param in model.middle_block.parameters():
                    param.grad = None
                    param.requires_grad = False

            # encoder blocks
            # Note: encoder is in the order fine to coarse
            # Note: len(model.input_blocks) == 12, if 4 channel multipliers and model.num_res_blocks == 2, since 1 input block, and downsample blocks at every resolution except for the coarsest
            # Note: we never freeze all layers, hence we don't need to account for the +1 input block (see forward)
            upper_range = (n_levels_used_training-1) * (model.num_res_blocks+1) - 1 if n_levels_used_training > 1 else (n_levels_used_training-1) * (model.num_res_blocks+1)  # accounting for coarsest resolution not having downsamling layer
            for input_block in model.input_blocks[::-1][:upper_range]: 
                print("j {}, freeze Enc . : in_blo {}".format(j, input_block[0].__class__.__name__))  # taking 0-th layer of Sequential object
                for param in input_block.parameters(): 
                    param.grad = None
                    param.requires_grad = False

            # always freeze very first input block (which doesn't have any parameters)
            print("j {}, freeze Enc . : 'very first input block' {}".format(j, model.input_blocks[0].__class__.__name__))  # taking 0-th layer of Sequential object
            for param in model.input_blocks[0].parameters(): 
                param.grad = None
                param.requires_grad = False

            # time embedding (!!!)
            for level in list(range(model.n_levels))[::-1][:n_levels_used_training - 1]: 
                print("j {}, freeze Tim ! : level {}. time_emb {}".format(j, level, model.time_embed_list[level].__class__.__name__))
                for param in model.time_embed_list[level].parameters(): 
                    param.grad = None
                    param.requires_grad = False
        

        # print("Starting training on resolution {} (j = {})...".format(res, j))
        for cur_it in range(num_iters):  # using train_iter for the training iteration index

            # TODO to(device) call necessary here? isn't this already done above? 
            if H.DATASET == 'mnist':
                batch_x = next(repeat_data_iter)[0].to(H.DEVICE)
            elif H.DATASET == 'mnist_triangular': 
                batch_x = next(repeat_data_iter).to(H.DEVICE)  
            elif H.DATASET == 'celeba':
                batch_x = next(repeat_data_iter).to(H.DEVICE)

            # downsample batch_x
            if SEQU_TRAIN_ALGO: 
                n_downsample = int(math.log2(dataset_res // cur_res))
                if n_downsample > 0: 
                    xfm = DWTForward(J=n_downsample, mode='zero', wave='haar').to(H.DEVICE)
                    yl, _ = xfm(batch_x)
                    ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
                    yl_inv = ifm((yl, []))
                    # before the above 10 lines of downsampling, batch_x is in [-1,1] (by computing min and max over a batch)
                    # now, after the DWT forward and inverse, x_0 is no longer normalized (e.g. is now in [-8.0,+6.7961])
                    # hence normalize batch_x back to [-1,1]
                    batch_x = yl_inv / math.pow(2, n_downsample)  # correct scaling to ensure we are in the original data range
                else: 
                    # do not downsample batch_x
                    pass

            # sample time indices
            if H.STAGED_PARTITIONED_TIME_INTERVALS: 
                t = diffusion.sample_t(batch_x, stage=j, n_stages=len(H.NUM_ITERATIONS_LIST))  
            else: 
                # regular time
                t = diffusion.sample_t(batch_x)  

            # print("Stage {}, t = {}".format(j, t))

            # forward of data by t timesteps --> produces x_t
            perturbed_sample = diffusion.sample_x(batch_x, t)
            x_t = perturbed_sample.x_t

            # first add noise then downsample order - change (1/3) - comment in, and comment out the batch_x downsampling above
            # downsample the x_t (not the batch_x) --> order matters!!! 
            # if SEQU_TRAIN_ALGO: 
            #     n_downsample = int(math.log2(dataset_res // cur_res))
            #     if n_downsample > 0: 
            #         xfm = DWTForward(J=n_downsample, mode='zero', wave='haar').to(H.DEVICE)
            #         yl, _ = xfm(x_t)
            #         ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
            #         yl_inv = ifm((yl, []))
            #         # before the above 10 lines of downsampling, x_t is in [-1,1] (by computing min and max over a batch)
            #         # now, after the DWT forward and inverse, x_0 is no longer normalized (e.g. is now in [-8.0,+6.7961])
            #         # hence normalize x_t back to [-1,1]
            #         x_t = yl_inv / math.pow(2, n_downsample)  # correct scaling to ensure we are in the original data range
            #     else: 
            #         # do not downsample x_t
            #         pass

            # noise prediction
            # note: model_out is list if model=='unet_wavelet'
            # print("n_levels_used_training: ", n_levels_used_training)
            model_out, _ = model(x_t, t.unsqueeze(-1), n_levels_used=n_levels_used_training)  # model receives x_t and t as input

            optimizer.zero_grad()

            if H.MULTI_RES_LOSS:
                noise_orig = perturbed_sample.z
                # compute downsampled noise, in the same order as in decoder (coarsest to finest)
                noise_list = []  
                for k in list(range(0, model.n_levels))[::-1]:  # reversing the list
                    
                    # if using the sequential algorithm: already downsampled above, do less here.
                    if SEQU_TRAIN_ALGO: 
                        # first add noise then downsample order - change (2/3)
                        # below must be k = k
                        k = k - n_downsample  # - n_downsample  # noise not anymore "downsampled" via x_t "automatically"

                    if k > 0:  
                        xfm = DWTForward(J=k, mode='zero', wave='haar').to(H.DEVICE)
                        yl, _ = xfm(noise_orig)
                        ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
                        yl_inv = ifm((yl, []))

                        # before the above lines of downsampling, noise_orig is in a certain range, and DWTForward changes the scale by a factor 2^J
                        # hence normalize yl_inv back to the original range
                        yl_inv = yl_inv / math.pow(2, k)  # correct scaling to ensure we are in the original data range

                        noise_list.append(yl_inv)
                    elif k == 0: 
                        noise_list.append(noise_orig)  # on highest res, use the original noise
                    # if k < 0: no noise appended

                # output
                noise = noise_list

            # single-res case 
            else: 
                noise = perturbed_sample.z

                # first add noise then downsample order - change (3/3)
                # n_downsample = int(math.log2(dataset_res // cur_res))
                # if n_downsample > 0:
                #     xfm = DWTForward(J=n_downsample, mode='zero', wave='haar').to(H.DEVICE)
                #     yl, _ = xfm(noise)
                #     ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
                #     yl_inv = ifm((yl, []))
                #     # scale appropriately
                #     noise = yl_inv / math.pow(2, n_downsample)  # correct scaling to ensure we are in the original data range
                
            # if MULTI_RES_SEQU_MODE: 
            #     assert H.MODEL == 'unet'  # only U-Net supported at the moment
                
            #     # Version 1: loss at stage i of training algorithm is loss on res i
            #     # produce noise 
            #     noise_orig = perturbed_sample.z
            #     dwt_j = model.n_levels - 1 - j
            #     print("dwt_j: ", dwt_j, noise_orig.shape)
            #     xfm = DWTForward(J=dwt_j, mode='zero', wave='haar').to(H.DEVICE)
            #     yl, _ = xfm(noise_orig)
            #     ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
            #     yl_inv = ifm((yl, []))

            #     noise = yl_inv

                # Version 2: loss at stage i of training algorithm is sum_i loss on res i
                # TODO
                # TODO note that this requires to use the G-Net, and likewise requires to merge the G-Net and U-Net class somehow

            # if H.MODEL == 'unet_wavelet':
            #     assert len(model_out) == len(noise)
            #     for k, (model_out_k, noise_k) in enumerate(zip(model_out, noise)): 
            #         print("res_" + str(model_output_k.shape[2]), "model_output: ", model_out_k.shape, "noise: ", noise_k.shape)
            # else: 
            #     print("model_output: ", model_out.shape, "noise: ", noise.shape)

            # if H.MULTI_RES_LOSS: 
            #     for (o, n) in zip(model_out, noise):
            #         print("BEFORE LOSS: model_output: ", o.shape, "noise: ", n.shape)
            # else: 
            #     print("BEFORE LOSS: model_output: ", model_out.shape, "noise: ", noise.shape)

            # in particular, not on very first iteration (not required here)
            if SEQU_TRAIN_ALGO and H.MULTI_RES_LOSS and H.LOSS_LINEAR_LOOP_IN and j != 0: 
                # linear schedule for K iterations after each new loss is looped in
                K = int(.2 * num_iters)
                last_loss_schedule_weight = np.min(np.array([cur_it / K, 1.])).item()
            else: 
                # no schedule
                last_loss_schedule_weight = 1.

            # print("last_loss_schedule_weight: ", last_loss_schedule_weight)
            loss, loss_list = diffusion.loss(model_output=model_out, noise=noise, last_loss_schedule_weight=last_loss_schedule_weight)
            
            loss.backward()
            optimizer.step()    

            if train_iter % H.TRAIN_METRICS_EVERY_ITERS == 0:
                log_dict = {}
                # log_dict['train/loss_' + str(j)] = loss  # .detach().cpu().numpy().item()
                log_dict['train/loss'] = loss  # .detach().cpu().numpy().item()
                wandb.log(log_dict, step=train_iter)

                if H.MULTI_RES_LOSS:
                    log_dict = {}
                    for k, l in enumerate(loss_list): 
                        log_dict['train/res_' + str(resolutions[k]) + '_loss'] = l
                    wandb.log(log_dict, step=train_iter)

            
            # every N iterations: do plotting
            if (train_iter % H.SAMPLES_EVERY_ITERS == 0) or (H.SAMPLES_EVERY_ITERS_FINAL_RES != -1 and j == len(H.NUM_ITERATIONS_LIST) - 1 and train_iter % H.SAMPLES_EVERY_ITERS_FINAL_RES == 0):
                print("Sampling at iter :", train_iter, "---")
                for r in resolutions: 
                    # print("Sampling on res = ", r)
                    
                    # dec_idx is in reverse order of level
                    n_levels_used_sampling = res_to_n_levels_used[r]
                    # print("n_levels_used_sampling: ", n_levels_used_sampling)
 
                    N_SAMPLES = 25
                    x_T = torch.randn((N_SAMPLES, in_channels, r, r), device=H.DEVICE)

                    x_0, x_mean = diffusion.reverse_sample(x_T, model, n_levels_used=n_levels_used_sampling)

                    # FOR TESTING PURPOSES ONLY
                    # if r == 4: 
                    #     # timesteps = torch.linspace(T, eps, N).to(x_T.device)  # where T and epsilon are used
                    #     # x, x_mean = (x_T, x_T)
                    #     t = 1   # timesteps[-1]
                    #     B = N_SAMPLES
                    #     vec_t = (torch.ones(B).to(x_T.device) * t).reshape(B, 1)
                    #     # print("TESTING --- bef model")
                    #     out = model(x_T_4_test, vec_t, n_levels_used=1)
                    #     # print("TESTING --- aft model")

                    #     norm_out = torch.norm(out.flatten())
                    #     print("!!! TESTING OF SAMPLING ON j={}, RES=4: model norm of out = {:.10f}, {}".format(j, norm_out, torch.isclose(norm_out, last_norm_out, atol=1e-08, rtol=0.) )  )
                    #     last_norm_out = norm_out
                    
                    # plot
                    plot_x = x_mean
                    # plot_x *= 255.

                    # if H.DATASET == 'celeba': 
                    #     # unnormalize the image
                    #     plot_x = unnormalize_fn(plot_x, norm_mean, norm_std)
                    #     # convert to int
                    #     # plot_x = plot_x.int()

                    # unnormalize "manually", now back in [0, 1]
                    # plot_x = (plot_x + 1) / 2.

                    # TODO this is a hack!!!!!!!
                    # normalise plot_x to [0, 1]
                    # plot_x = (plot_x - plot_x.min()) / (plot_x.max() - plot_x.min())

                    if H.to_square_preprocess: 
                        # print("swapping started")
                        plot_x = plot_x.cpu().detach().numpy()
                        plot_x_list = []
                        for x in plot_x: 
                            #swap_array(img,tri_array,square_array, 'cubic').reshape(n,n)
                            plot_x_list.append(swap_array(x.squeeze(), preprocessor.tri_array, preprocessor.square_array, 'cubic').reshape(2**(preprocessor.J),2**(preprocessor.J)))
                        plot_x = torch.tensor(plot_x_list).unsqueeze(1)  # insert channel dim back in
                        plot_x = plot_x.cpu().detach()
                        # print("swapping done")

                    # .permute(2, 0, 1)
                    # print(plot_x)
                    # cutting padding away
                    if H.to_square_preprocess:
                        img_list = [plot_x[i] for i in range(N_SAMPLES)]  # [:, 2:30, 2:30]
                    else: 
                        img_list = [plot_x[i].cpu().detach() for i in range(N_SAMPLES)]  # [:, 2:30, 2:30]

                    fig, _ = plot_uncond_samples(img_list=img_list, uncond_samples_n_rows=int(math.sqrt(N_SAMPLES)), uncond_samples_n_cols=int(math.sqrt(N_SAMPLES)))
                    wandb.log({'samples/res_' + str(r): wandb.Image(plt)}, step=train_iter)
                    plt.close(fig=fig)  # close the figure

                    # plot histogramme of plot_x values
                    plt.clf()
                    fig = plt.figure()
                    plt.hist(plot_x.cpu().detach().numpy().flatten())
                    wandb.log({'histogramme_samples/res_' + str(r): wandb.Image(plt)}, step=train_iter)
                    plt.close(fig=fig)  # close the figure


            if H.U_NET_NORM and (train_iter % H.U_NET_NORM_EVERY_ITERS == 0): 
                with torch.no_grad():
                    # initialize dataset and loader
                    if H.DATASET == 'mnist' or H.DATASET == 'mnist_triangular':
                        ds = MNIST(load=('MNIST' in os.listdir('./datasets')), num_channels=1, device = H.DEVICE)
                    elif H.DATASET == 'celeba':
                        dataset_train, dataset_val, dataset_test, norm_mean, norm_std = get_celeba_datasets()
                        # TODO for now, just use the train set
                        ds = dataset_train

                    # data to GPU
                    # TODO for now, just training data to GPU
                    ds.data = ds.data.to(H.DEVICE)

                    # scale original data down to the desired lower resolution res, by repeatedly applying average pooling
                    for _ in range(int(math.log2(dataset_res // res))):
                        print("downsampling")
                        ds.data = torch.nn.functional.avg_pool2d(ds.data, kernel_size=(2, 2), stride=(2, 2))
                    dataloader = DataLoader(ds, batch_size=H.BATCH_SIZE, drop_last=True)

                    repeat_data_iter_plot = repeater(dataloader)  # TODO why necessary? 

                    norms_list, t_list = [], []

                    for _ in range(H.U_NET_NORM_N_BATCHES):   

                        # TODO to(device) call necessary here? isn't this already done above? 
                        if H.DATASET == 'mnist' or H.DATASET == 'mnist_triangular':
                            batch_x = next(repeat_data_iter_plot)[0].to(H.DEVICE)
                        elif H.DATASET == 'celeba':
                            batch_x = next(repeat_data_iter_plot).to(H.DEVICE)

                        # downsample the batch_x 
                        if SEQU_TRAIN_ALGO: 
                            n_downsample = int(math.log2(dataset_res // cur_res))
                            if n_downsample > 0: 
                                xfm = DWTForward(J=n_downsample, mode='zero', wave='haar').to(H.DEVICE)
                                yl, _ = xfm(batch_x)
                                ifm = DWTInverse(mode='zero', wave='haar').to(H.DEVICE)
                                yl_inv = ifm((yl, []))
                                # before the above 10 lines of downsampling, batch_x is in [-1,1] (by computing min and max over a batch)
                                # now, after the DWT forward and inverse, x_0 is no longer normalized (e.g. is now in [-8.0,+6.7961])
                                # hence normalize batch_x back to [-1,1]
                                batch_x = yl_inv / math.pow(2, n_downsample)  # correct scaling to ensure we are in the original data range
                            else: 
                                # do not downsample batch_x
                                pass

                        # sample time indices
                        t = diffusion.sample_t(batch_x)  

                        # forward of data by t timesteps --> produces x_t
                        perturbed_sample = diffusion.sample_x(batch_x, t)
                        x_t = perturbed_sample.x_t

                        # noise prediction
                        # note: model_out is list if model=='unet_wavelet'
                        # print("n_levels_used_training: ", n_levels_used_training)
                        model_out, norms = model(x_t, t.unsqueeze(-1), n_levels_used=n_levels_used_training, u_net_norm=True)  # u_net_norm=True (since already inside case); model receives x_t and t as input
                        norms_list.append(norms)
                        t_list.append(t)

                    fig = plot_unet_norms(norms_list=norms_list, t_list=t_list)
                    wandb.log({'plot_unet_norms': wandb.Image(plt)}, step=train_iter)
                    plt.close(fig=fig)  # close the figure


            # if SEQ_TRAIN_ALGO == True: only sample once the model is fully trained, i.e. at the very last iter
            if H.DO_SUPERRES and ((SEQU_TRAIN_ALGO and train_iter % (sum(H.NUM_ITERATIONS_LIST) - 1) == 0 and train_iter != 0) or (not SEQU_TRAIN_ALGO and H.MULTI_RES_LOSS and train_iter % H.SUPERRES_SAMPLES_EVERY_ITERS == 0)):
                print("Superresolution sampling at iter :", train_iter, "---")
                
                # superres configs
                SOURCE_TARGET_RES_TUPLES = [(4, 32), (8, 32), (16, 32)]
                N_NOISE_SAMPLES = 10
                N_SUPERRES_SAMPLES_PER_NOISE = 2

                for res_tuple in SOURCE_TARGET_RES_TUPLES: 
                    assert res_tuple[0] in all_possible_unet_resolutions 
                    assert res_tuple[1] in all_possible_unet_resolutions


                
                for (source_res, target_res) in SOURCE_TARGET_RES_TUPLES:
                    superres_n_res = int(math.log2(target_res / source_res))
                    print("Superresolution sampling: source_res = ", source_res, "target_res = ", target_res, "superres_n_levels = ", superres_n_res)

                    img_list = []
                    x_T = torch.randn((N_NOISE_SAMPLES, in_channels, source_res, source_res), device=H.DEVICE)

                    # lower-res image
                    x_0, x_mean = diffusion.reverse_sample(x_T, model, n_levels_used=n_levels_used_sampling)  # without superresolution
                    # include the image into higher resolution by upsampling
                    plot_x = x_mean
                    up_layer = UpInterpolate(up_rate=int(2 ** superres_n_res))
                    plot_x = up_layer(plot_x)
                    img_list += [plot_x[i].cpu().detach() for i in range(N_NOISE_SAMPLES)]

                    for _ in range(N_SUPERRES_SAMPLES_PER_NOISE): 

                        # upsample x_T
                        up_layer = UpInterpolate(up_rate=int(2 ** superres_n_res))
                        x_T_up = up_layer(x_T)

                        x_0, x_mean = diffusion.reverse_sample(x_T_up, model, n_levels_used=n_levels_used_sampling + superres_n_res)
                        plot_x = x_mean

                        # downsample back
                        # for _ in range(superres_n_res): 
                        #     plot_x = torch.nn.functional.avg_pool2d(plot_x, kernel_size=2, stride=2)

                        img_list += [plot_x[i].cpu().detach() for i in range(N_NOISE_SAMPLES)]  # [:, 2:30, 2:30]

                    # plotting
                    fig, _ = plot_uncond_samples(img_list=img_list, uncond_samples_n_rows=N_SUPERRES_SAMPLES_PER_NOISE, uncond_samples_n_cols=N_NOISE_SAMPLES)
                    wandb.log({'superres/s_' + str(source_res) + '_t_' + str(target_res): wandb.Image(plt)}, step=train_iter)
                    plt.close(fig=fig)  # close the figure

                    



            # Models, optimizer and scheduler saving 'checkpoint'
            if train_iter % H.ITERS_PER_MODEL_SAVE == 0 and not H.ITERS_PER_MODEL_SAVE == -1:  # -1 means not saving model
                # print("saving!!!")
                prefix = "iter-%d-"%(train_iter)
                # Note: not saving diffusion, because H is loaded and then restored diffusion just as before. diffusion object has no parameters.
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, prefix + "model.th"))
                # torch.save(model_eval.state_dict(), os.path.join(wandb.run.dir, prefix + "model_eval.th"))
                torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, prefix + "optimizer.th"))
                # torch.save(scheduler.state_dict(), os.path.join(wandb.run.dir, prefix + "scheduler.th"))
                torch.save(train_iter, os.path.join(wandb.run.dir, "last_save_iter.th"))  # save without prefix

            
            # increment iterator
            train_iter += 1

            # print("train_iter: ", train_iter)



        # freeze parameters of model
        # for param in model.parameters():
        #     param.requires_grad = False


        # TODO dataset to lower resolution
        # TODO once trained: initialise class MultiResolutionDiffusion(Diffusion):


if __name__ == '__main__':
    main()