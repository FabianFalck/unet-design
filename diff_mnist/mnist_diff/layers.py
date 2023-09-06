
import torch
from torch_ddpm.ddpm.models.basic.layers import MLP
from torch_ddpm.ddpm.models.basic.time_embedding import get_timestep_embedding
from torch_ddpm.ddpm.models.unet.unet import SiLU

from torch_ddpm.ddpm.models.unet.layers import normalization, zero_module, conv_nd






class ScoreNetwork(torch.nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=2):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2  # x_t and t as input
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        # self.out = torch.nn.Sequential(
        #     # normalization(1),
        #     SiLU(),
        #     zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        # )
        self.out = torch.nn.Identity()


    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) > 2:
            # convert image to two dimensions 
            is_image = True
            image_shape = x.shape
            x = torch.reshape(x, (x.shape[0], -1))
        else: 
            is_image = False

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 

        if is_image: 
            out = torch.reshape(out, image_shape)
            # normalize in [0,1]  # TODO only valid for datasets like MNIST!!!
            out = self.out(out)

        return out



# def build_fc_network(layer_dims, activation="relu", dropout_prob=0.):
#     """
#     Stacks multiple fully-connected layers with an activation function and a dropout layer in between.
#     - Source used as orientation: https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py
#     Args:
#         layer_dims: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the input
#                     and output dimension of the ith layer, respectively.
#         activation: Activation function to choose. "relu" or "sigmoid".
#         dropout_prob: Dropout probability between every fully connected layer with activation.
#     Returns:
#         An nn.Sequential object of the layers.
#     """
#     # Note: possible alternative: OrderedDictionary
#     net = []
#     for i in range(1, len(layer_dims)):
#         net.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
#         if activation == "relu":
#             net.append(nn.ReLU())
#         elif activation == "sigmoid":
#             net.append(nn.Sigmoid())
#         elif activation == "elu":
#             net.append(nn.ELU())
#         net.append(nn.Dropout(dropout_prob))
#     net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

#     return net



class UpInterpolate(torch.nn.Module):
    def __init__(self, up_rate):
        super().__init__()
        # F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        self.up_rate = up_rate

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.up_rate, mode='nearest')

        return x