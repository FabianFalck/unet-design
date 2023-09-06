import os

import numpy as np
import torch
import torchvision.transforms as transforms
from .two_dim import get_two_dim_ds


def data_transform(d_config, X):
    if d_config.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    elif d_config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if d_config.rescaled:
        X = 2 * X - 1.0
    elif d_config.logit_transform:
        X = logit_transform(X)

    return X


def inverse_data_transform(d_config, X):

    if d_config.logit_transform:
        X = torch.sigmoid(X)
    elif d_config.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
