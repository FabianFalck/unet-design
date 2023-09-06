import math
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import TensorDataset
import torchvision
import os
import wandb
import yaml
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from tqdm import tqdm
import sklearn

from hyperparams import args_parser

from model import Unetbase_G

from evaluation import getDSC_pt, getAVD_pt, getHausdorff_pt, getLesionDetection_pt

import plotting
from plotting import plot_segmentation

from utils import convertfloat32touint8

from model import DWTBlock


def load_dict_from_yaml(file_path):
    """
    Load args from .yml file.
    """
    with open(file_path) as file:
        yaml_dict = yaml.safe_load(file)

    # create argsparse object
    # parser = argparse.ArgumentParser(description='MFCVAE training')
    # args, unknown = parser.parse_known_args()
    # for key, value in config.items():
    #     setattr(args, key, value)

    # return args.__dict__  # return as dict, not as argsparse
    return yaml_dict


# a Dataset class which manages images and masks
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        super().__init__()
        assert len(images) == len(masks)  # make sure we pass the same number of images and masks (this was a bug for 2 days during the development and obviously deteriorated training)
        self.images = torch.from_numpy(images)
        self.masks = torch.from_numpy(masks)
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        # data augmentation
        if self.transform is not None: 
            # apply transform separately for each channel as transform only supports 1 and 3 channels
            image[0, ...] = self.transform(torch.unsqueeze(image[0, ...], 0))
            image[1, ...] = self.transform(torch.unsqueeze(image[1, ...], 0))

        mask = self.masks[index]

        return image, mask

    def __len__(self):
        return len(self.images)


def freeze_layers(stage, model):
        
        n_levels_used = stage + 1  # since we are in staged training mode
        # down
        for i in list(range(model.n_levels))[-n_levels_used+1:]: 
            print("freeze down i", i)
            for param in model.down[i].parameters(): 
                param.grad = None
                param.requires_grad = False
        # up 
        for j in list(range(n_levels_used - 1)): 
            print("freeze up j", j)
            for param in model.up[j].parameters(): 
                param.grad = None
                param.requires_grad = False
        # head
        for k in range(model.n_levels - n_levels_used + 1, model.n_levels): 
            print("freeze head k", k)
            for param in model.image_proj_list[k].parameters(): 
                param.grad = None
                param.requires_grad = False
        # tail
        for l in range(n_levels_used - 1): 
            print("freeze tail l", l)
            for param in model.final_list[l].parameters(): 
                param.grad = None
                param.requires_grad = False


def dice_coef_for_training_pt(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss_pt(y_true, y_pred):
    """See also: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183 """

    return 1. - dice_coef_for_training_pt(y_true, y_pred)



def evaluate(model, data_loader, val_or_test, iter, criterion, n_levels_used_evaluate): 
    print(val_or_test + " at iter: ", iter, " ...")

    thresholds = np.arange(0.1, 1, 0.1).tolist()  # thresholds used for evaluation metrics
    # round each threshold to 2 decimals, due to numerical instability
    thresholds = [round(threshold, 2) for threshold in thresholds]

    dsc_dict, precision_dict, recall_dict, f1_dict, confusion_matrix_dict, accuracy_dict = {}, {}, {}, {}, {}, {}
    for threshold in thresholds:
        dsc_dict[threshold] = []
        precision_dict[threshold] = []
        recall_dict[threshold] = []
        f1_dict[threshold] = []
        confusion_matrix_dict[threshold] = []
        accuracy_dict[threshold] = []
    loss_list = []
    output_flat_list, mask_flat_list = [], []  # used for computing precision-recall curve
    image_list, y_true_list, y_pred_list = [], [], []  # used to draw segmentation results
    n_batches_seg_plot = int(H.n_images_seg_to_plot / H.batch_size) + 1  # number of batches used for drawing the plots
    for batch_idx_val, (image, mask) in enumerate(data_loader):
        # put batch on right device
        image = image.to(H.device)
        mask = mask.to(H.device)

        # downsample 
        if SEQU_TRAIN_ALGO: 
            n_downsample = model.n_levels - n_levels_used_evaluate
            if n_downsample > 0: 
                with torch.no_grad(): 
                    dwt_1 = DWTBlock(J=n_downsample, out_channels=image.shape[1], mode=H.dwt_mode, wave=H.dwt_wave).to(H.device)
                    dwt_2 = DWTBlock(J=n_downsample, out_channels=mask.shape[1], mode=H.dwt_mode, wave=H.dwt_wave).to(H.device)
                    image = dwt_1(image)
                    mask = dwt_2(mask)

                    if SEQU_TRAIN_ALGO: 
                        # mask needs to be binarised again
                        # also for evaluation, we here use the threshold .5 which seems natural due to the averaging with haar wavelets
                        threshold = 0.0   # [0.3, 0.25, 0.2, 0.1][n_downsample]

                        mask = torch.where(mask > threshold, torch.ones_like(mask).int(), torch.zeros_like(mask).int())  # binarised  ; int conversion to catch potential numerical instabilities
            else: 
                # do not downsample batch_x
                pass

        output = model(image, n_levels_used=n_levels_used_evaluate)
        loss = criterion(mask, output)
        loss_list.append(loss.cpu().detach().numpy())

        output = output.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        if batch_idx_val < 20:  # number of batches used for drawing the plots
            output_flat = np.copy(output.flatten())  # need to copy such that thresholding below is not performed on this object
            mask_flat = np.copy(mask.flatten())  # need to copy such that thresholding below is not performed on this object
            output_flat_list.append(output_flat)
            mask_flat_list.append(mask_flat)

        if batch_idx_val < n_batches_seg_plot:  # number of batches used for drawing the plots
            image = image.cpu().detach().numpy()

            # convert to uint8
            # TODO note: normalization to [0, 255] is done per batch --> not correct, but works for now
            image_seg = convertfloat32touint8(image)
            chosen_modality = 0
            image_seg = np.expand_dims(image_seg[:, chosen_modality, ...], 1)  # only first modality is plotted
            # convert mask to boolean
            # y_true_seg = mask == 1.0
            y_true_seg = mask

            # convert to torch
            image_seg = torch.tensor(image_seg).clone()
            y_true_seg = torch.tensor(y_true_seg).clone()

            for j in range(image_seg.shape[0]): 
                image_list.append(image_seg[j])
            for j in range(y_true_seg.shape[0]):
                y_true_list.append(y_true_seg[j])

        # if batch_idx_val == 0:
        #     fig = plt.figure()
        #     display = PrecisionRecallDisplay.from_predictions(mask, output)  # , name="LinearSVC"
        #     display.plot(ax=plt.gca())
        #     wandb.log({'val/prec_recall_curve': wandb.Image(plt)}, step=iter)
        #     plt.close(fig)
        
        y_pred_of_best_dsc = None
        best_dsc = None

        

        for threshold in thresholds:
            

            # output to 0 or 1 (binary); threshold .5 chosen
            y_pred = np.where(output > threshold, np.ones_like(output).astype(int), np.zeros_like(output).astype(int))  # binarised  ; int conversion to catch potential numerical instabilities

            # convert to int
            # output = output.int()
            # mask = mask.int()

            if batch_idx_val < n_batches_seg_plot: 
                y_pred_seg = torch.tensor(y_pred).clone()
                # convert mask to boolean
                # y_pred_seg = y_pred_seg == 1.0
                y_pred_copy = y_pred_seg


            # remove channel dimension (dimension 1)
            y_pred = y_pred.squeeze(1)
            y_true = mask.squeeze(1)


            y_pred_flat = y_pred.flatten()
            y_true_flat = y_true.flatten()

            # for confusion matrix, assert that y_true_flat and y_pred_flat only contain 0 and 1
            assert np.all(np.isin(y_true_flat, [0, 1]))
            assert np.all(np.isin(y_pred_flat, [0, 1]))

            # validation metrics
            precision = sklearn.metrics.precision_score(y_true_flat, y_pred_flat)
            recall = sklearn.metrics.recall_score(y_true_flat, y_pred_flat)
            f1 = sklearn.metrics.f1_score(y_true_flat, y_pred_flat)
            confusion_matrix = sklearn.metrics.confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
            accuracy = sklearn.metrics.accuracy_score(y_true_flat, y_pred_flat)

            # convert to other type
            y_pred = sitk.GetImageFromArray(y_pred)
            y_true = sitk.GetImageFromArray(y_true)

            # convert to numpy, then to sitk format
            # for i in range(image.shape[0]):
                # must be done for each image separately, batch dimension not supported by stk (?)
                # one_mask = sitk.GetImageFromArray(mask_bin[i])
                # one_output = sitk.GetImageFromArray(output_bin[i])
                
                # TODO didnt' make any sense so far: we only computed it off the last slice!

                # validation metrics
            dsc = getDSC_pt(y_true, y_pred)
            # avd = getAVD_pt(mask_bin, output_bin)   # throws division by zero error 
            # h95 = getHausdorff_pt(mask_bin, output_bin)  # throws error
            # recall, precision, f1 = getLesionDetection_pt(y_true, y_pred)

            # find the prediction corresponding to the best dsc
            if batch_idx_val < n_batches_seg_plot and (best_dsc is None or dsc > best_dsc):
                y_pred_of_best_dsc = y_pred_copy

            # append
            # print("threshold: ", threshold)
            dsc_dict[threshold].append(dsc)
            precision_dict[threshold].append(precision)
            recall_dict[threshold].append(recall)
            f1_dict[threshold].append(f1)
            confusion_matrix_dict[threshold].append(confusion_matrix)

        if batch_idx_val < n_batches_seg_plot: 
            y_pred_seg = torch.tensor(y_pred_of_best_dsc).clone()
            for j in range(y_pred_seg.shape[0]):
                y_pred_list.append(y_pred_seg[j])

        # TODO JUST DEBUGGIN ONE ITERATION
        if H.debug_breaks and batch_idx_val == 2:  # use 3 batches during debug mode  
            break    

    output = np.concatenate(output_flat_list, axis=0)
    mask = np.concatenate(mask_flat_list, axis=0)

    # plot segmentation overlayed figure
    fig, grid_img = plot_segmentation(image_list=image_list, y_true_list=y_true_list, y_pred_list=y_pred_list, n_images_seg_to_plot=H.n_images_seg_to_plot)
    wandb.log({val_or_test + '/segmentation': wandb.Image(plt)}, step=iter)  #
    plt.close(fig=fig)


    # draw validation plots
    # plot histogramme of output
    # plotting
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(output)
    # xlim between [0,1]
    ax.set_xlim(0, 1)
    wandb.log({val_or_test + '/output_hist': wandb.Image(plt)}, step=iter)  # 
    # plt.show()
    plt.close(fig=fig)

    # precision recall curve
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    precision, recall, _ = precision_recall_curve(mask, output)  
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(color='black', ax=ax)
    # display.plot(ax=plt.gca())
    wandb.log({val_or_test + '/prec_recall_curve': wandb.Image(plt)}, step=iter)
    plt.close(fig=fig)


    # aggegate the metrics
    for threshold in thresholds:
        dsc_dict[threshold] = np.mean(dsc_dict[threshold])
        precision_dict[threshold] = np.mean(precision_dict[threshold])
        recall_dict[threshold] = np.mean(recall_dict[threshold])
        f1_dict[threshold] = np.mean(f1_dict[threshold])

        # sometimes throws an error for unknown reason. if so, catch it: 
        try: 
            confusion_matrix_dict[threshold] = np.sum(np.array(confusion_matrix_dict[threshold]), axis=0)  # takes sum of matrices over the list
        except:
            print("WARNING: could not aggregate confusion matrix. skipping.")
            confusion_matrix_dict[threshold] = np.array([[0,0],[0,0]])
        # convert to int matrix
        confusion_matrix_dict[threshold] = confusion_matrix_dict[threshold].astype(int)
        accuracy_dict[threshold] = np.mean(accuracy_dict[threshold])


    loss_mean = np.mean(loss_list)

    # plot confusion matrix
    for threshold in thresholds:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dict[threshold])
        disp.plot(ax=ax, values_format='d')  # display the int rather than in scientific notation
        wandb.log({val_or_test + '/' + str(threshold) + '/confusion_matrix': wandb.Image(plt)}, step=iter)
        plt.close(fig=fig)


    # create dictionary
    metrics_dict = {}
    # add eval loss
    metrics_dict[val_or_test + "/loss"] = loss_mean
    for threshold in thresholds:
        metrics_dict[val_or_test + "/" + str(threshold) + "/dsc"] = dsc_dict[threshold]
        metrics_dict[val_or_test + "/" + str(threshold) + "/precision"] = precision_dict[threshold]
        metrics_dict[val_or_test + "/" + str(threshold) + "/recall"] = recall_dict[threshold]
        metrics_dict[val_or_test + "/" + str(threshold) + "/f1"] = f1_dict[threshold]
        metrics_dict[val_or_test + "/" + str(threshold) + "/accuracy"] = accuracy_dict[threshold]
        
    # val log

    # TODO DEBUGGING
    # for key, value in metrics_dict.items():
        # wandb.log({key: value}, step=iter)
        
    wandb.log(metrics_dict, step=iter)

    return loss_mean


if __name__ == "__main__": 

    H = args_parser()

    # make entire code deterministic
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)

    # load wandb api key, project name and entity name
    wandb_config = load_dict_from_yaml('wandb.yml')
    # login to wandb and create run
    wandb.login(key=wandb_config['user'])
    wandb_run = wandb.init(project=wandb_config['project_name'], entity=wandb_config['team_name'], mode=H.wandb_mode)

    # load preprocessed data (produced by preprocessing.py)
    data_preprocess_path = 'data_preprocessed'
    train_images = np.load(os.path.join(data_preprocess_path, 'images_three_datasets_sorted_train.npy'))
    train_masks = np.load(os.path.join(data_preprocess_path, 'masks_three_datasets_sorted_train.npy'))
    test_images = np.load(os.path.join(data_preprocess_path, 'images_three_datasets_sorted_test.npy'))
    test_masks = np.load(os.path.join(data_preprocess_path, 'masks_three_datasets_sorted_test.npy'))

    # insert dimension 3 in masks (which just has 1)
    train_masks = np.expand_dims(train_masks, axis=3)
    test_masks = np.expand_dims(test_masks, axis=3)
    # resort dimensions such that (samples, channels, rows, cols)
    train_images = train_images.transpose(0, 3, 1, 2)
    train_masks = train_masks.transpose(0, 3, 1, 2)
    test_images = test_images.transpose(0, 3, 1, 2)
    test_masks = test_masks.transpose(0, 3, 1, 2)

    # normalise images s.t. mean=0, std=1, based on mean and std of training set
    # TODO separately for each modality?
    # separately for each modality
    for m in range(2): # 2 modalities
        norm_mean = np.mean(train_images[:, m].flatten())
        norm_std = np.std(train_images[:, m].flatten())
        train_images[:, m, :, :] = (train_images[:, m, :, :] - norm_mean) / norm_std
        test_images[:, m, :, :] = (test_images[:, m, :, :] - norm_mean) / norm_std

    # select indices
    fraction = 0.1
    n_val_patients_per_site= int(math.ceil(fraction * 20))  # 20 patients per cite
    n_images_per_patient_0_1 = 48   # 0,1 site
    n_images_per_patient_2 = 83  # 2 site
    assert n_images_per_patient_0_1 * 40 + n_images_per_patient_2 * 20 == train_images.shape[0]
    val_indices = list(range(0, n_val_patients_per_site * n_images_per_patient_0_1)) + \
                    list(range(20*n_images_per_patient_0_1, 20*n_images_per_patient_0_1 + n_val_patients_per_site * n_images_per_patient_0_1)) + \
                    list(range(40*n_images_per_patient_0_1, 40*n_images_per_patient_0_1 + n_val_patients_per_site * n_images_per_patient_2))
    train_indices = list(set(range(train_images.shape[0])) - set(val_indices))
    assert len(set(train_indices).intersection(set(val_indices))) == 0  # check that train_indices and val_indices are not overlapping
    # slice data correctly
    train_images = train_images[train_indices]
    train_masks = train_masks[train_indices]
    val_images = train_images[val_indices]
    val_masks = train_masks[val_indices]
    
    
    # data augmentation  [none, auto, manual]
    # Version 1: AutoAugment
    if H.data_augmentation == 'auto':
        transforms = torchvision.transforms.AutoAugment()
    # Version 2: rotation, flipping, shear, zoom
    elif H.data_augmentation == 'manual1':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(360),  # 90 degrees due to horizontal and vertical flipping (probably 45 is enough)
            # torchvision.transforms.RandomHorizontalFlip(),  # not needed due to rotation
            # torchvision.transforms.RandomVerticalFlip(),  # not needed due to rotation
            torchvision.transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),  # what is shear? : see https://hasty.ai/docs/mp-wiki/augmentations/shear
        ])
    elif H.data_augmentation == 'manual2':
        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomRotation(360),  # 90 degrees due to horizontal and vertical flipping (probably 45 is enough)
            torchvision.transforms.RandomHorizontalFlip(),  # not needed due to rotation
            torchvision.transforms.RandomVerticalFlip(),  # not needed due to rotation
            # torchvision.transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        ])
    elif H.data_augmentation == 'manual3':
        # Y version
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomAffine(degrees=0, shear=(18, 18), scale=(0.9, 1.1)),  # what is shear? : see https://hasty.ai/docs/mp-wiki/augmentations/shear

        ])
    # Version 3: no augmentation
    elif H.data_augmentation == 'none':
        transforms = None
    else: 
        raise ValueError('which_augmentation value not valid')

    # instantiate dataset
    train_data = Dataset(train_images, train_masks, transform=transforms)
    val_data = Dataset(val_images, val_masks)
    test_data = Dataset(test_images, test_masks)
    # take fraction of training data for validation
    # take fraction of the number of patients in each of the three sites


    

    # instantiate dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=H.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=H.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=H.batch_size, shuffle=True)

    # load mdoel 
    model = Unetbase_G(
                    dwt_encoder=H.dwt_encoder,
                    up_fct = H.up_fct, 
                    n_extra_resnet_layers = H.n_extra_resnet_layers,
                    multi_res_loss = H.multi_res_loss,
                    sequ_mode = len(H.num_epochs_list) > 1,
                    hidden_channels = H.hidden_channels,
                    no_skip_connection=H.no_skip_connection, 
                    no_down_up = H.no_down_up,
                    dwt_mode = H.dwt_mode,
                    dwt_wave = H.dwt_wave,
                ).to(H.device)
    
    # count total number of parameters and print
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)  #  ; 7754180
    params_string = f'{params_count:,}'
    print("Number of parameters of model: " + params_string)
    # store number of parameters count in configs
    H.n_params = params_string

    # define loss function
    criterion = dice_coef_loss_pt
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=H.lr)

    wandb.config.update(H)
    wandb.watch(model)

    print("Starting the training loop...")

    # train
    iter = 0
    epoch = 0
    metric_early_stop = 100000000
    early_stop = False
    count_no_improvement = 0
    # several more useful variables
    SEQU_TRAIN_ALGO = len(H.num_epochs_list) > 1
    dataset_res = train_data.images[0].shape[2]


    for stage in range(len(H.num_epochs_list)):

        if SEQU_TRAIN_ALGO and H.freeze_lower_res and stage != 0: 
            print("FREZING LAYERS in stage %d!"%stage)
            freeze_layers(stage=stage, model=model)
        
        # sequential training: admin variables
        if SEQU_TRAIN_ALGO: 
            resolutions = [dataset_res // (2 ** i) for i in range(model.n_levels)]
            all_possible_unet_resolutions = resolutions.copy()
            res_to_n_levels_used = {res: model.n_levels - int(math.log2(dataset_res / res)) for res in resolutions}
            cur_res = resolutions[-(stage+1)]
            if SEQU_TRAIN_ALGO: 
                # only consider the highest resolutions at this stage
                resolutions = resolutions[(-stage)-1:]
        else: 
            resolutions = [dataset_res]
            cur_res = dataset_res
            res_to_n_levels_used = {dataset_res: model.n_levels}

        for epoch_in_stage in tqdm(range(H.num_epochs_list[stage])):
            for batch_idx, (image, mask) in enumerate(train_loader):

                # sequential training
                if SEQU_TRAIN_ALGO:
                    n_levels_used_training = stage + 1
                else: 
                    n_levels_used_training = model.n_levels

                # put batch on right device
                image = image.to(H.device)
                mask = mask.to(H.device)

                # downsample 
                if SEQU_TRAIN_ALGO: 
                    n_downsample = int(math.log2(dataset_res // cur_res))
                    if n_downsample > 0: 
                        with torch.no_grad(): 
                            dwt_1 = DWTBlock(J=n_downsample, out_channels=image.shape[1], mode=H.dwt_mode, wave=H.dwt_wave).to(H.device)
                            dwt_2 = DWTBlock(J=n_downsample, out_channels=mask.shape[1], mode=H.dwt_mode, wave=H.dwt_wave).to(H.device)
                            image = dwt_1(image)
                            mask = dwt_2(mask)

                            # mask needs to be binarised again
                            # for training, we here use the threshold .5 which seems natural 
                            threshold = 0.5
                            mask = torch.where(mask > threshold, torch.ones_like(mask).int(), torch.zeros_like(mask).int())  # binarised  ; int conversion to catch potential numerical instabilities
                    else: 
                        # do not downsample batch_x
                        pass

                optimizer.zero_grad()
                output = model(image, n_levels_used=n_levels_used_training)
                loss = criterion(mask, output)  # mask: torch.Size([30, 1, 200, 200]) ; output: torch.Size([30, 1, 200, 200])
                loss.backward()
                optimizer.step()

                # train log
                if iter % H.train_loss_every_iters == 0: 
                    wandb.log({'train/loss': loss.item()}, step=iter)
                    
                if iter % H.train_hist_every_iters == 0 or iter % H.train_prec_recall_curve_every_iters == 0: 
                    output = output.cpu().detach().numpy().flatten()
                    mask = mask.cpu().detach().numpy().flatten()

                if iter % H.train_hist_every_iters == 0:
                    # plot histogramme of output
                    # plotting
                    plt.clf()
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.hist(output)
                    ax.set_xlim(0, 1)
                    wandb.log({'train/output_hist': wandb.Image(plt)}, step=iter)  # 
                    # plt.show()
                    plt.close(fig=fig)

                if iter % H.train_prec_recall_curve_every_iters == 0:   
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    precision, recall, _ = precision_recall_curve(mask, output)  
                    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
                    disp.plot(color='black', ax=ax)
                    # display.plot(ax=plt.gca())
                    wandb.log({'train/prec_recall_curve': wandb.Image(plt)}, step=iter)
                    plt.close(fig)

                # validation procedure ---> done every epoch (not here)
                # if iter % H.val_every_iters == 0 and iter > 0: 
                    
                    
                # print("iter: %d, loss: %f" % (iter, loss.item()))

                # increment iteration
                iter += 1

                # TODO JUST DEBUGGIN ONE ITERATION
                if H.debug_breaks: 
                    break

            # validation procedure every epoch
            if epoch_in_stage % H.val_every_epochs == 0: 
                val_loss = evaluate(model=model, data_loader=val_loader, val_or_test='val', iter=iter, criterion=criterion, n_levels_used_evaluate=n_levels_used_training)  # use the level currently trained to evaluate loss

                # early stopping
                # assumes early stop metric is lower is better
                if val_loss < metric_early_stop - H.early_stop_min_improvement:   # at least 0.001 improvement
                    save_path = os.path.join(wandb.run.dir, 'model.pt')
                    print("Validation loss improved from %f to %f. Saving model at path "  % (metric_early_stop, val_loss) + save_path + " ...")
                    torch.save(model.state_dict(), save_path)

                    metric_early_stop = val_loss
                    count_no_improvement = 0
                else: 
                    count_no_improvement += 1
                    if H.early_stop_patience != -1 and count_no_improvement > H.early_stop_patience:
                        print("Early stopping!!! No improvement for %d validation precedures." % H.early_stop_patience)
                        early_stop = True
                        break
            
            wandb.log({'epoch': epoch}, step=iter)
            epoch += 1

   

            # TODO JUST DEBUGGIN ONE ITERATION
            if H.debug_breaks: 
                break
        
        if early_stop: 
            break


    
    # test procedure
    print("Training complete.")
    # print("Reinitialise model and load best model params...")
    # model = Unetbase_G(
    #                 dwt_encoder=H.dwt_encoder,
    #                 up_fct = H.up_fct, 
    #                 n_extra_resnet_layers = H.n_extra_resnet_layers,
    #                 multi_res_loss = H.multi_res_loss,
    #                 sequ_mode = len(H.num_epochs_list) > 1,
    #                 hidden_channels = H.hidden_channels,
    #                 no_skip_connection=H.no_skip_connection, 
    #                 no_down_up = H.no_down_up,
    #                 dwt_mode = H.dwt_mode,
    #                 dwt_wave = H.dwt_wave,
    #             ).to(H.device)
    load_path = os.path.join(wandb.run.dir, 'model.pt')
    print("Load best model params from path " + load_path + " ...")
    model.load_state_dict(torch.load(load_path))
    print("Testing the best model model...")
    evaluate(model=model, data_loader=test_loader, val_or_test='test', iter=iter, criterion=criterion, n_levels_used_evaluate=model.n_levels)  # use all levels during testing (all have been trained)
    print("Testing complete.")

