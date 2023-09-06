from torchvision import datasets, transforms
from torchvision.utils import draw_segmentation_masks
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def plot_grid(img_list, n_rows, n_cols):  # , y_labels, y_labels_2, fontsize=3
    # find dimensions of one input
    img_dims = img_list[0].size()

    # make grid of images
    # make_grid expects list of images, each of shape (C x H x W)
    # nrow is number of images per row
    grid_img = make_grid(img_list, nrow=n_cols, pad_value=0,  # 0 is black padding
                                                        padding=1,   # 0 is no padding
                                                        scale_each=True)  # TODO is this correct? 

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

    # for r, y_label in enumerate(y_labels):
    #     ax.text(-0.01, 1 - (1/len(y_labels)) * r - (1/len(y_labels)) * .5 , y_label, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=fontsize, rotation=90)
    # for r, y_label in enumerate(y_labels_2):
    #     ax.text(-0.05, 1 - (1/len(y_labels)) * r * 2 - 1 * (1/len(y_labels)), y_label, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=fontsize, rotation=90)

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    return fig, grid_img


def draw_segmentation_mask_onto_image(image, mask, color): 
    """
    Overlay segmentation mask onto image
    see https://stackoverflow.com/questions/44535068/opencv-python-cover-a-colored-mask-over-a-image#comment76061213_44535068
    """
    mask = mask.squeeze()


    if color == 'red': 
        image[0, :, :][mask == 1.] = 255
        image[1, :, :][mask == 1.] = 0
        image[2, :, :][mask == 1.] = 0
    elif color == 'green':
        image[0, :, :][mask == 1.] = 0
        image[1, :, :][mask == 1.] = 255
        image[2, :, :][mask == 1.] = 0    
    elif color == 'blue': 
        image[0, :, :][mask == 1.] = 0
        image[1, :, :][mask == 1.] = 0
        image[2, :, :][mask == 1.] = 255

    return image




def plot_segmentation(image_list, y_true_list, y_pred_list, n_images_seg_to_plot=10): 
    """
    y_true are the masks (ground truth)  
    y_pred are the predictions, binarised with a certain threshold
    """
    assert len(image_list) == len(y_true_list) == len(y_pred_list)
    assert image_list[0].shape[2:] == y_true_list[0].shape[2:] == y_pred_list[0].shape[2:]
    assert n_images_seg_to_plot <= len(image_list) * image_list[0].shape[0]

    # select right number of images
    image_list = image_list[:n_images_seg_to_plot]
    y_true_list = y_true_list[:n_images_seg_to_plot]
    y_pred_list = y_pred_list[:n_images_seg_to_plot]

    # TODO comment back in when doing overlay onto MRI image !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # convert grayscale images to RGB by repeating the gray-scale channel 3 times
    if image_list[0].shape[0] == 1:
        image_list = [image.repeat(3, 1, 1) for image in image_list]

    plot_list = []
    for (image, y_true, y_pred) in zip(image_list, y_true_list, y_pred_list): 
        # image_y_true = draw_segmentation_masks(image, y_true, alpha=1., colors='blue')
        # image_y_pred = draw_segmentation_masks(image, y_pred, alpha=1., colors='blue')

        # oerlay segmentation 
        image_y_true = draw_segmentation_mask_onto_image(image.clone(), y_true, 'green')
        image_y_pred = draw_segmentation_mask_onto_image(image.clone(), y_pred, 'blue')


        # append images
        # plot_list.append(image)  # original image (without an overlayed mask)
        plot_list.append(image_y_true)
        plot_list.append(image_y_pred)
    
    # plot the grid
    fig, grid_img = plot_grid(plot_list, n_rows=n_images_seg_to_plot, n_cols=10)

    # plt.show()
    
    return fig, grid_img
   