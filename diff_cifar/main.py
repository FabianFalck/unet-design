import copy
import json
import os
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import numpy as np

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNetWaveletEnc  # UNet, 
from score.both import get_inception_and_fid_score
import wandb
from utils import load_dict_from_yaml, download_some_wandb_files
import math
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse

from model import UpSample, DownSample
from hyperparams import args_parser, check_hyperparams

global FLAGS
FLAGS = args_parser()
# check hyperparams for consistency etc.
check_hyperparams(FLAGS)



# load wandb api key, project name and entity name
wandb_config = load_dict_from_yaml('wandb.yml')
# login to wandb and create run
wandb.login(key=wandb_config[FLAGS.user])
wandb_run = wandb.init(project=wandb_config['project_name'], entity=wandb_config['team_name'], mode=FLAGS.wandb_mode)

# make entire code deterministic
np.random.seed(FLAGS.SEED)
torch.manual_seed(FLAGS.SEED)
torch.cuda.manual_seed(FLAGS.SEED)


# compute today's date and the current time, and return as a string
def get_timestamp():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%m-%d_%H-%M-%S")






def ema(source, target, decay):
    # Version 1
    # source_dict = source.state_dict()
    # target_dict = target.state_dict()
    # for key in source_dict.keys():
    #     # only update the parameters which require gradients or are not frozen
    #     # otherwise, frozen parameters will converge towards the parameter values in the source model (here: net_model)
    #     # Option 1: 
    #     # if source_dict[key].requires_grad: 
    #     # Option 2:  
    #     # if source_dict[key].grad is not None: 
    #     target_dict[key].data.copy_(
    #         target_dict[key].data * decay +
    #         source_dict[key].data * (1 - decay))

    # Version 2
    for (param_source, param_target) in zip(source.parameters(), target.parameters()):
        if param_source.grad is not None: 
            param_target.data.copy_(
                param_target.data * decay +
                param_source.data * (1 - decay))


        # if not source_dict[key].data.requires_grad and target_dict[key].data.requires_grad: 
            # print("debug")


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            # use all levels for images generated for evaluation
            batch_images = sampler(x_T.to(device), n_levels_used=model.n_levels).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # auxiliary / remembering variables
    RESTORE = FLAGS.TRAIN_ID is not None or FLAGS.TEST_ID is not None
    training = FLAGS.TEST_ID is None  # either restored from training or training from scratch 
    train_iter = FLAGS.TRAIN_ITER
    test_iter = FLAGS.TEST_ITER

    # download some files first and load them
    if RESTORE:
        files_to_restore = ["H.dict", "last_save_iter.th"]  # 'last_save_iter.th' is just storing an int
        run_id = FLAGS.TRAIN_ID if FLAGS.TRAIN_ID is not None else FLAGS.TEST_ID
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=run_id)
        # Note: loads another H dictionary in the case of restoring which overwrites the new one above
        flags = torch.load(os.path.join(wandb.run.dir, files_to_restore[0]))  # overwrites H parsed above
        last_save_iter = torch.load(os.path.join(wandb.run.dir, files_to_restore[1]))
        # In the restored H, we overwrite train or test restore information which we need below
        if training:
            FLAGS.TRAIN_ID = run_id
            FLAGS.TRAIN_ITER = train_iter
        else:
            FLAGS.TEST_ID = run_id
            FLAGS.TEST_ITER = test_iter
        print("Note: Restoring run " + run_id + ". Any passed command line arguments are ignored!")   # Note: Could even throw exception if this is the case.

    if FLAGS.TRAIN_ID is not None:
        train_iter = last_save_iter if FLAGS.TRAIN_ITER is None else FLAGS.TRAIN_ITER
        restore_iter = train_iter
        model_load_file_name = 'iter-%d-model.th'%train_iter
        ema_model_load_file_name = 'iter-%d-ema_model.th'%train_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%train_iter
        scheduler_load_file_name = 'iter-%d-scheduler.th'%train_iter
        files_to_restore = [model_load_file_name, ema_model_load_file_name, optimizer_load_file_name, scheduler_load_file_name]  
        download_run_id = FLAGS.TRAIN_ID
        print("Continuing training at iter %d"%step)
    elif FLAGS.TEST_ID is not None:
        test_iter = last_save_iter if FLAGS.TEST_ITER is None else FLAGS.TEST_ITER
        restore_iter = test_iter
        # Note: could only load model_eval here
        model_load_file_name = 'iter-%d-model.th'%test_iter
        ema_model_load_file_name = 'iter-%d-ema_model.th'%test_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%test_iter
        scheduler_load_file_name = 'iter-%d-scheduler.th'%test_iter
        files_to_restore = [model_load_file_name, ema_model_load_file_name, optimizer_load_file_name, scheduler_load_file_name]  
        download_run_id = FLAGS.TEST_ID
    else:
        train_iter = 0

    if RESTORE:
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=download_run_id)

    # save the hyperparam config after everything was set up completely (some values of H will only be written at run-time, but everything is written at this point)
    if not RESTORE:
        print("saving H config...")
        torch.save(FLAGS, os.path.join(wandb.run.dir, "H.dict"))

    # assign train_iter to step, because that's the variable we use going forward
    step = train_iter

    

    # ---
    

    # load checkpoint
    # if FLAGS.load_dir != '':
        # ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        # # straight away use loaded FLAGS
        # FLAGS = ckpt['FLAGS']
        # # create new logdir
        # FLAGS.logdir = './logs/' + get_timestamp()

    

    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    print("FLAGS.logdir: ", FLAGS.logdir)

    # model setup
    # if FLAGS.model == 'unet':
        # net_model = UNet(
        #     T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        #     num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # elif FLAGS.model == 'unet_wavelet_enc':
    net_model = UNetWaveletEnc(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, dwt_encoder=FLAGS.DWT_ENCODER, multi_res_loss=FLAGS.MULTI_RES_LOSS, downsample_type=FLAGS.DOWNSAMPLE_TYPE)
    
    if len(FLAGS.NUM_ITERATIONS_LIST) > 1: 
        SEQU_TRAIN_ALGO = True
        assert net_model.n_levels == len(FLAGS.NUM_ITERATIONS_LIST)
        print("Multi-resolution mode!")
    else: 
        SEQU_TRAIN_ALGO = False
        print("Single resolution mode!")

    ema_model = copy.deepcopy(net_model)

    if RESTORE: 
        net_model.load_state_dict(torch.load(os.path.join(wandb.run.dir, model_load_file_name)))
        ema_model.load_state_dict(torch.load(os.path.join(wandb.run.dir, ema_model_load_file_name)))



    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, multi_res_loss=FLAGS.MULTI_RES_LOSS, sequ_train_algo=SEQU_TRAIN_ALGO, device=FLAGS.device).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type, multi_res_loss=FLAGS.MULTI_RES_LOSS).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type, multi_res_loss=FLAGS.MULTI_RES_LOSS).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    # os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T_dict = {}
    highest_res = FLAGS.img_size  # CIFAR10-specific ; 
    all_res = [highest_res // (2 ** i) for i in range(net_model.n_levels)]
    for r in all_res: 
        x_T = torch.randn(FLAGS.sample_size, 3, r, r)
        x_T = x_T.to(device)
        x_T_dict[r] = x_T
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    # writer = SummaryWriter(FLAGS.logdir)
    # writer.add_image('real_sample', grid)
    # writer.flush()
    # # backup all arguments
    # with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
    #     f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # if FLAGS.load_dir != '':
    #     step = ckpt['step']
    # else:
        # initialise new step
    # step = 0   # global step   # we do this now above already  #  TODO delete eventually

    # compute dataset min and max (used for normalization below)
    dataset_min, dataset_max = dataset.data.min(), dataset.data.max()

    assert FLAGS.T > 1  # otherwise very nasty error
    
    # for testing purposes
    # last_norm_out_ema, last_norm_out_net = torch.tensor(-1.), torch.tensor(-1.)

    for j, num_iters in enumerate(FLAGS.NUM_ITERATIONS_LIST): 

        # just for testing purposes
        # if j == 3:   # res 32 start
        #     # freeze ALL parameters
        #     for p, param in enumerate(net_model.parameters()):  
        #         if p != 0:  # do not freeze very first parameter, as otherwise error is thrown
        #             param.requires_grad = False
        #             param.grad = None

        # slightly weird condition
        if SEQU_TRAIN_ALGO or FLAGS.MULTI_RES_LOSS: 
            # some variables for sampling
            # small difference to multiresdiff implementation: assumes we can only train the model on the highest resolution of the dataset)
            resolutions = [highest_res // (2 ** i) for i in range(net_model.n_levels)]
            all_possible_unet_resolutions = resolutions.copy()
            res_to_n_levels_used = {res: net_model.n_levels - int(math.log2(highest_res/res)) for res in resolutions}
            cur_res = resolutions[-(j+1)]
            if SEQU_TRAIN_ALGO: 
                # only consider the highest resolutions at this stage
                resolutions = resolutions[(-j)-1:]
        else: 
            resolutions = [highest_res]
            cur_res = highest_res
            res_to_n_levels_used = {highest_res: net_model.n_levels}
        

        if SEQU_TRAIN_ALGO:
            n_levels_used_training = j + 1
        else: 
            n_levels_used_training = net_model.n_levels

        print("--- cur_res = {} (j = {})...".format(cur_res, j))

        # freeze parameters of all except the finest level used in decoder in this stage of the sequential training algorithm)
        if FLAGS.FREEZE_LOWER_RES: 
            # print("printing all layers")
            # for level in range(net_model.n_levels): 
            #     for layer in net_model.upblocks[level]:
            #         print("level {}. layer {}".format(level, layer.__class__.__name__))
            #     for layer in net_model.tail_list[level]:
            #         print("level {}. layer {}".format(level, layer.__class__.__name__))
            #     for layer in net_model.downblocks[level]:
            #         print("level {}. layer {}".format(level, layer.__class__.__name__))

            assert SEQU_TRAIN_ALGO
            # decoder blocks
            idx_freeze = 0
            for level in list(range(net_model.n_levels))[::-1][:n_levels_used_training - 1]: 
                for layer in net_model.upblocks[level]:
                    if not (isinstance(layer, UpSample) and idx_freeze == n_levels_used_training - 2):   # do not freeze 'final' UpSample layer 
                        print("j {}, freeze Dec # : level {}. layer {}.".format(j, level, layer.__class__.__name__))  #  Norm over all parameters {} ;;;  torch.norm(torch.cat([t.flatten() for t in list(layer.parameters())]))
                        # compute norm over all parameters of layer
                        for param in layer.parameters():
                            param.grad = None  # param.requires_grad = False
                            param.requires_grad = False
                for param in net_model.tail_list[level].parameters():
                    print("j {}, freeze Tai - : level {}.".format(j, level))  # First param norm {} ;;; , torch.norm(torch.cat([t.flatten() for t in list(layer.parameters())]))
                    param.grad = None
                    param.requires_grad = False

                idx_freeze += 1

            # middle blocks (part of decoder)
            if n_levels_used_training >= 2: 
                for layer in net_model.middleblocks:
                    print("j {}, freeze Mid % : layer {}.".format(j, layer.__class__.__name__))   # .  Norm over all parameters {} ;;; , torch.norm(torch.cat([t.flatten() for t in list(layer.parameters())]))
                    for param in layer.parameters():
                        param.grad = None
                        param.requires_grad = False
     
            
            # encoder blocks
            # Note: encoder is in the order fine to coarse
            # Note: len(model.input_blocks) == 12, if 4 channel multipliers and model.num_res_blocks == 2, since 1 input block, and downsample blocks at every resolution except for the coarsest
            # Note: we never freeze all layers, hence we don't need to account for the +1 input block (see forward)
            idx_freeze_2 = 0
            for level in list(range(net_model.n_levels))[::-1][:n_levels_used_training - 1]: 
                for layer in net_model.downblocks[level]:
                    # if j == 2: 
                        # print("hallo")
                    if not (isinstance(layer, DownSample) and idx_freeze_2 == n_levels_used_training - 1):   # do not freeze 'first' DownSample layer
                        print("j {}, freeze Enc . : level {}. layer {}.".format(j, level, layer.__class__.__name__))  #   Norm over all parameters {} ;;; , torch.norm(torch.cat([t.flatten() for t in list(layer.parameters())]))
                        # print("n_levels_used_training = {}".format(n_levels_used_training))
                        for param in layer.parameters(): 
                            param.grad = None
                            param.requires_grad = False
                
                idx_freeze_2 += 1

            # time embedding (!!!)
            for level in list(range(net_model.n_levels))[::-1][:n_levels_used_training - 1]: 
                for param in net_model.time_embedding_list[level].parameters():
                    print("j {}, freeze Tim ! : level {}.".format(j, level))  #   Norm over all parameters {} ;;; , torch.norm(torch.cat([t.flatten() for t in list(layer.parameters())])) 
                    param.grad = None
                    param.requires_grad = False


        # new optimizer and scheduler in every stage
        # this also makes each stage to have its own warmup which is probably beneficial
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

        if RESTORE: 
            optim.load_state_dict(torch.load(os.path.join(wandb.run.dir, optimizer_load_file_name)))
            sched.load_state_dict(torch.load(os.path.join(wandb.run.dir, scheduler_load_file_name)))

        # weights&biases tracking (gradients, network topology)
        # only watch model (this is the one doing the training step)
        # only do so when training
        if FLAGS.TEST_ID is None:
            wandb.watch(net_model)

        # print out flags, just to have it on the console
        print(FLAGS)

        wandb.config.update(FLAGS)

        # start training
        print("Start training...")
        with trange(num_iters, dynamic_ncols=True) as pbar:
            for stage_step in pbar:
                # train
                optim.zero_grad()
                x_0 = next(datalooper).to(device)

                # downsample the batch_x 
                if SEQU_TRAIN_ALGO: 
                    n_downsample = int(math.log2(highest_res // cur_res))
                    if n_downsample > 0: 
                        xfm = DWTForward(J=n_downsample, mode='zero', wave='haar').to(FLAGS.device)
                        yl, _ = xfm(x_0)
                        ifm = DWTInverse(mode='zero', wave='haar').to(FLAGS.device)
                        yl_inv = ifm((yl, []))
                        x_0 = yl_inv

                        # before the above 10 lines of downsampling, batch_x is in [-1,1] (by computing min and max over a batch)
                        # now, after the DWT forward and inverse, x_0 is no longer normalized (e.g. is now in [-8.0,+6.7961])
                        # hence normalize batch_x back to [-1,1]
                        x_0 = x_0 / math.pow(2, n_downsample)  # correct scaling to ensure we are in the original data range

                    else: 
                        # do not downsample batch_x
                        pass
                else: 
                    n_downsample = 0

                loss, loss_list = trainer(x_0, n_levels_used=n_levels_used_training, n_downsample=n_downsample)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                # only select the first NUM_SAMPLES_VIS images
                NUM_SAMPLES_VIS = 25
                assert x_0.shape[0] >= NUM_SAMPLES_VIS
                x_0 = x_0[:NUM_SAMPLES_VIS]

                # log
                # wandb.log({'loss': loss}, step=step)
                # writer.add_scalar('loss', loss, step)
                pbar.set_postfix(loss='%.3f' % loss)

                if step % FLAGS.TRAIN_METRICS_EVERY_ITERS == 0:
                    log_dict = {}
                    # log_dict['train/loss_' + str(j)] = loss  # .detach().cpu().numpy().item()
                    log_dict['train/loss'] = loss  # .detach().cpu().numpy().item()
                    wandb.log(log_dict, step=step)

                    if FLAGS.MULTI_RES_LOSS:
                        log_dict = {}
                        for k, l in enumerate(loss_list): 
                            log_dict['train/res_' + str(resolutions[k]) + '_loss'] = l
                        wandb.log(log_dict, step=step)


                # sample
                if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:  
                    net_model.eval()
                    print("Sampling at iter :", step, "---")
                    for r in resolutions: 
                        print("Sampling on res = ", r, "---")
                        n_levels_used_sampling = res_to_n_levels_used[r]
                        with torch.no_grad():
                            # Note: we use here the same random noise sample again and again; this could be changed

                            # TESTING PURPOSES 
                            # if r == 4: 
                            #     time_step = 1
                            #     x_t = x_T_dict[r]
                            #     t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                            #     out = ema_sampler.model(x_t, t, n_levels_used=1)
                            #     # flatten out and print its norm
                            #     norm_out = torch.norm(out.flatten())
                            #     print("!!! TESTING OF SAMPLING ON j={}, RES=4: ema-model norm of out = {:.10f}, {}".format(j, norm_out, torch.isclose(norm_out, last_norm_out_ema, atol=1e-08, rtol=0.) )  )
                            #     last_norm_out_ema = norm_out

                            #     out = net_model(x_t, t, n_levels_used=1)
                            #     norm_out = torch.norm(out.flatten())
                            #     print("!!! TESTING OF SAMPLING ON j={}, RES=4: net_model norm of out = {:.10f}, {}".format(j, norm_out, torch.isclose(norm_out, last_norm_out_net, atol=1e-08, rtol=0.) )  )
                            #     last_norm_out_net = norm_out

                            #     # torch.manual_seed(0)
                            #     x_0 = ema_sampler(x_T_dict[4], n_levels_used=res_to_n_levels_used[4])
                            #     print("!!! TESTING OF SAMPLING ON j={}, RES=4: norm of x_0 (sample) = ".format(j), torch.norm(x_0.flatten()))

                           


                            x_0 = ema_sampler(x_T_dict[r], n_levels_used=n_levels_used_sampling)
                            x_0 = (x_0 + 1.) / 2.  # normalize [-1,1] to [0,1]
                            grid = make_grid(x_0, normalize=True)

                            # log grid 
                            plt.clf()
                            n_images = int(math.sqrt(x_0.shape[0]))
                            fig = plt.figure(figsize=(n_images, n_images))
                            ax = fig.add_subplot(1, 1, 1)
                            # permute dimensions of image from (channels, width, height) to (width, height, channels)
                            grid = grid.permute(1, 2, 0)
                            ax.imshow(grid.cpu().detach().numpy())  # all other datasets
                            ax.set_xticks([])
                            ax.set_yticks([])
                            fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
                            wandb.log({'samples/res_' + str(r): wandb.Image(plt)}, step=step)
                            plt.close(fig=fig)  # close the figure

                            # plot histogramme of plot_x values
                            plt.clf()
                            fig = plt.figure()
                            plt.hist(x_0.cpu().detach().numpy().flatten())
                            wandb.log({'histogramme_samples/res_' + str(r): wandb.Image(plt)}, step=step)
                            plt.close(fig=fig)  # close the figure

                            # path = os.path.join(
                            #     FLAGS.logdir, 'sample', '%d.png' % step)
                            # save_image(grid, path)
                            # writer.add_image('sample', grid, step)
                    net_model.train()

                # save
                
                    
                    # ckpt = {
                    #     'net_model': net_model.state_dict(),
                    #     'ema_model': ema_model.state_dict(),
                    #     'sched': sched.state_dict(),
                    #     'optim': optim.state_dict(),
                    #     'step': step,
                    #     'x_T': x_T_dict[highest_res],
                    #     # 'FLAGS': FLAGS,
                    # }
                    # torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

                if step > 0 and FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                    print("Saving at step :", step, "---")
                    prefix = "iter-%d-"%(step)
                    torch.save(net_model.state_dict(), os.path.join(wandb.run.dir, prefix + "model.th"))
                    torch.save(ema_model.state_dict(), os.path.join(wandb.run.dir, prefix + "ema_model.th"))
                    torch.save(optim.state_dict(), os.path.join(wandb.run.dir, prefix + "optimizer.th"))
                    torch.save(sched.state_dict(), os.path.join(wandb.run.dir, prefix + "scheduler.th"))
                    torch.save(step, os.path.join(wandb.run.dir, "last_save_iter.th"))  # save without prefix



                # evaluate
                if step > 0 and FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0 and (not(FLAGS.MULTI_RES_LOSS) or step >= int(sum(FLAGS.NUM_ITERATIONS_LIST[:-1]))):                    
                    print("Evaluating FID at step :", step, "---")
                    # the samples generated here are disjunct from the ones used for sampling above (in this new implementation)
                    net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                    ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                    metrics = {
                        'IS': net_IS[0],
                        'IS_std': net_IS[1],
                        'FID': net_FID,
                        'IS_EMA': ema_IS[0],
                        'IS_std_EMA': ema_IS[1],
                        'FID_EMA': ema_FID
                    }
                    wandb.log(metrics, step=step)


                step += 1

                # pbar.write(
                #     "%d/%d " % (step, FLAGS.total_steps) +
                #     ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                # for name, value in metrics.items():
                #     writer.add_scalar(name, value, step)
                # writer.flush()
                # with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                #     metrics['step'] = step
                #     f.write(json.dumps(metrics) + "\n")

        # writer.close()


def eval():
    # auxiliary / remembering variables
    RESTORE = FLAGS.TRAIN_ID is not None or FLAGS.TEST_ID is not None
    training = FLAGS.TEST_ID is None  # either restored from training or training from scratch 
    step = FLAGS.TRAIN_ITER
    test_iter = FLAGS.TEST_ITER

    # download some files first and load them
    if RESTORE:
        files_to_restore = ["H.dict", "last_save_iter.th"]  # 'last_save_iter.th' is just storing an int
        run_id = FLAGS.TRAIN_ID if FLAGS.TRAIN_ID is not None else FLAGS.TEST_ID
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=run_id)
        # Note: loads another H dictionary in the case of restoring which overwrites the new one above
        flags = torch.load(os.path.join(wandb.run.dir, files_to_restore[0]))  # overwrites H parsed above
        last_save_iter = torch.load(os.path.join(wandb.run.dir, files_to_restore[1]))
        # In the restored H, we overwrite train or test restore information which we need below
        if training:
            FLAGS.TRAIN_ID = run_id
            FLAGS.TRAIN_ITER = train_iter
        else:
            FLAGS.TEST_ID = run_id
            FLAGS.TEST_ITER = test_iter
        print("Note: Restoring run " + run_id + ". Any passed command line arguments are ignored!")   # Note: Could even throw exception if this is the case.

    if FLAGS.TRAIN_ID is not None:
        train_iter = last_save_iter if FLAGS.TRAIN_ITER is None else FLAGS.TRAIN_ITER
        restore_iter = train_iter
        model_load_file_name = 'iter-%d-model.th'%train_iter
        ema_model_load_file_name = 'iter-%d-ema_model.th'%train_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%train_iter
        scheduler_load_file_name = 'iter-%d-scheduler.th'%train_iter
        files_to_restore = [model_load_file_name, ema_model_load_file_name, optimizer_load_file_name, scheduler_load_file_name]  
        download_run_id = FLAGS.TRAIN_ID
    elif FLAGS.TEST_ID is not None:
        test_iter = last_save_iter if FLAGS.TEST_ITER is None else FLAGS.TEST_ITER
        restore_iter = test_iter
        # Note: could only load model_eval here
        model_load_file_name = 'iter-%d-model.th'%test_iter
        ema_model_load_file_name = 'iter-%d-ema_model.th'%test_iter
        optimizer_load_file_name = 'iter-%d-optimizer.th'%test_iter
        scheduler_load_file_name = 'iter-%d-scheduler.th'%test_iter
        files_to_restore = [model_load_file_name, ema_model_load_file_name, optimizer_load_file_name, scheduler_load_file_name]  
        download_run_id = FLAGS.TEST_ID
    else:
        step = 0

    if RESTORE:
        download_some_wandb_files(files_to_restore=files_to_restore, run_id=download_run_id)

    # save the hyperparam config after everything was set up completely (some values of H will only be written at run-time, but everything is written at this point)
    if not RESTORE:
        print("saving H config...")
        torch.save(FLAGS, os.path.join(wandb.run.dir, "H.dict"))

    print("Testing model at iteration", test_iter)

    # ---


    # model setup

    # old syntax / classes # TODO delete
    # model = UNet(
    #     T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
    #     num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # sampler = GaussianDiffusionSampler(
    #     model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
    #     mean_type=FLAGS.mean_type, var_type=FLAGS.var_type, multi_res_loss=FLAGS.MULTI_RES_LOSS).to(device)

    model = UNetWaveletEnc(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, dwt_encoder=FLAGS.DWT_ENCODER, multi_res_loss=FLAGS.MULTI_RES_LOSS)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type, multi_res_loss=FLAGS.MULTI_RES_LOSS).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate  # TODO old syntax delete
    # ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    # model.load_state_dict(ckpt['net_model'])
    assert RESTORE  # must be true 
    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, model_load_file_name)))
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    wandb.log({"net_model/IS": IS, "net_model/IS_std": IS_std, "net_model/FID": FID}, step=restore_iter)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples.png'),
    #     nrow=16)

    # Ema model
    # model.load_state_dict(ckpt['ema_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, ema_model_load_file_name)))
    wandb.log({"ema_model/IS": IS, "ema_model/IS_std": IS_std, "ema_model/FID": FID}, step=restore_iter)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))


    

    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples_ema.png'),
    #     nrow=16)


def main():
    # suppress annoying inception_v3 initialization warning
    global device

    # print("ATTN", FLAGS.attn)

    training = FLAGS.TEST_ID is None  # either restored from training or training from scratch 

    device = torch.device(FLAGS.device)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if training:
        train()
    else:
        eval()
    # else:
    #     raise ValueError('Neither train() nor eval() was called.')
        # print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    # app.run(main)
    # main(sys.argv)
    main()
