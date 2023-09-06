# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from torch import nn
import math

from .activations import ACTIVATION_REGISTRY

from pytorch_wavelets import DWTForward, DWTInverse


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if norm:
            # Original used BatchNorm2d
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.up(x1)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        return h


class Unetbase(nn.Module):
    """Our interpretation of the original U-Net architecture.

    Uses [torch.nn.GroupNorm][] instead of [torch.nn.BatchNorm2d][]. Also there is no `BottleNeck` block.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation="gelu",
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        self.image_proj = ConvBlock(insize, n_channels, activation=activation)

        self.down = nn.ModuleList(
            [
                Down(n_channels, n_channels * 2, activation=activation),
                Down(n_channels * 2, n_channels * 4, activation=activation),
                Down(n_channels * 4, n_channels * 8, activation=activation),
                Down(n_channels * 8, n_channels * 16, activation=activation),
            ]
        )
        self.up = nn.ModuleList(
            [
                Up(n_channels * 16, n_channels * 8, activation=activation),
                Up(n_channels * 8, n_channels * 4, activation=activation),
                Up(n_channels * 4, n_channels * 2, activation=activation),
                Up(n_channels * 2, n_channels, activation=activation),
            ]
        )
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        self.final = nn.Conv2d(n_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        h = self.image_proj(x)

        x1 = self.down[0](h)
        x2 = self.down[1](x1)
        x3 = self.down[2](x2)
        x4 = self.down[3](x3)
        x = self.up[0](x4, x3)
        x = self.up[1](x, x2)
        x = self.up[2](x, x1)
        x = self.up[3](x, h)

        x = self.final(x)
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

class FullResnetConvBlock(ConvBlock):
    def forward(self, x):
        h = super().forward(x)
        return h + x
    

class PartialResnetConvBlock(ConvBlock):
    """
    Use this module if the in and output channels are not the same.
    """
    def forward(self, x):
        h = self.activation(self.norm1(self.conv1(x)))  # changing the channels
        h = h + self.activation(self.norm2(self.conv2(h)))  # in_channels == out_channels -> can be ResNet
        return h
    

class DWTBlock(nn.Module): 
    def __init__(self, J, out_channels, mode='zero', wave='haar') -> None:
        super().__init__()

        self.J = J
        self.xfm = DWTForward(J=J, mode=mode, wave=wave)
        self.ifm = DWTInverse(mode=mode, wave=wave)
        self.out_channels = out_channels

    def forward(self, x):
        if self.J == 0: 
            # return identity, but with correct number of channels
            # Version 1: pass image itself on J == 0 
            out = x.repeat(1, int(self.out_channels / x.shape[1]) + 1, 1, 1)[:, :self.out_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
        else: 
            Yl, Yh = self.xfm(x)
            Yl_1_inv = self.ifm((Yl, []))   
            out = Yl_1_inv

            # before the above lines of downsampling, x is in a certain range, and DWTForward changes the scale by a factor 2^J
            # hence normalize batch_x back to the original range
            out = out / math.pow(2, self.J)  # correct scaling to ensure we are in the original data range

            # copy out across channel dim s.t. number of out channels is out_channels
            if x.shape[1] != self.out_channels:   # in_channels != out_channels
                out = out.repeat(1, int(self.out_channels / out.shape[1]) + 1, 1, 1)[:, :self.out_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
            
            assert out.shape[1] == self.out_channels

        return out






class Down_G(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu", dwt_encoder=False, no_down_up=False, dwt_mode='zero', dwt_wave='haar') -> None:
        super().__init__()
        if dwt_encoder:
            self.down = DWTBlock(J=1, out_channels=out_channels, mode=dwt_mode, wave=dwt_wave) if not no_down_up else DWTBlock(J=0, out_channels=out_channels, mode=dwt_mode, wave=dwt_wave)
        else: 
            self.conv = PartialResnetConvBlock(in_channels, out_channels, num_groups, norm, activation)
            self.pool = nn.AvgPool2d(2) if not no_down_up else nn.Identity()
        self.dwt_encoder = dwt_encoder

    def forward(self, x: torch.Tensor, finest_level: bool = False):
        if self.dwt_encoder:
            h = self.down(x)
        else: 
            if not finest_level: 
                h = self.pool(x)
            h = self.conv(h)

        return h


class Up_G(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu", up_fct='interpolate_nearest', n_extra_resnet_layers=0, no_skip_connection=False, no_down_up = False) -> None:
        super().__init__()
        if up_fct == 'conv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) if not no_down_up else nn.Identity()
        elif up_fct == 'interpolate_nearest':
            self.up_conv_channel_dim = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv = PartialResnetConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_fct = up_fct
        # extra ResNet layers, if adding parameters back in
        self.resnet_list = nn.ModuleList([FullResnetConvBlock(out_channels, out_channels, num_groups, norm, activation) for _ in range(n_extra_resnet_layers)])
        self.no_skip_connection = no_skip_connection
        self.no_down_up = no_down_up

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, finest_level: bool = False):
        if not finest_level: 
            if self.up_fct == 'conv': 
                h = self.up(x1)
            elif self.up_fct == 'interpolate_nearest':
                h = self.up_conv_channel_dim(x1)
                h = torch.nn.functional.interpolate(h, scale_factor=2) if not self.no_down_up else h  # identity when no on same res
        if self.no_skip_connection:
            x2 = torch.zeros_like(x2)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        if len(self.resnet_list) > 0:
            for resnet in self.resnet_list:
                h = resnet(h)  # does the h = h + x inside the forward
        return h


class Unetbase_G(nn.Module):
    """Our interpretation of the original U-Net architecture.

    Uses [torch.nn.GroupNorm][] instead of [torch.nn.BatchNorm2d][]. Also there is no `BottleNeck` block.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation="gelu",

        dwt_encoder = False, 
        up_fct = 'interpolate_nearest', 
        n_extra_resnet_layers = 0, 
        multi_res_loss = False,
        sequ_mode = False,
        no_skip_connection = False, 
        no_down_up = False, 
        dwt_mode = 'zero', 
        dwt_wave='haar'

    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        
        self.dwt_encoder = dwt_encoder
        self.up_fct = up_fct
        self.n_extra_resnet_layers = n_extra_resnet_layers
        self.multi_res_loss = multi_res_loss
        self.sequ_mode = sequ_mode
        self.no_skip_connection = no_skip_connection
        self.no_down_up = no_down_up
        self.dwt_mode = dwt_mode
        self.dwt_wave = dwt_wave
        


        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels

        # head
        self.image_proj_list = nn.ModuleList([])

        down_in_channels = [n_channels, n_channels * 2, n_channels * 4, n_channels * 8]
        self.down = nn.ModuleList(
            [
                Down_G(down_in_channels[0], n_channels * 2, activation=activation, dwt_encoder=dwt_encoder, no_down_up=self.no_down_up, dwt_mode=self.dwt_mode, dwt_wave=self.dwt_wave),
                Down_G(down_in_channels[1], n_channels * 4, activation=activation, dwt_encoder=dwt_encoder, no_down_up=self.no_down_up, dwt_mode=self.dwt_mode, dwt_wave=self.dwt_wave),
                Down_G(down_in_channels[2], n_channels * 8, activation=activation, dwt_encoder=dwt_encoder, no_down_up=self.no_down_up, dwt_mode=self.dwt_mode, dwt_wave=self.dwt_wave),
                Down_G(down_in_channels[3], n_channels * 16, activation=activation, dwt_encoder=dwt_encoder, no_down_up=self.no_down_up, dwt_mode=self.dwt_mode, dwt_wave=self.dwt_wave),
            ]
        )
        up_out_channels = [n_channels * 8, n_channels * 4, n_channels * 2, n_channels]
        self.up = nn.ModuleList(
            [
                Up_G(n_channels * 16, up_out_channels[0], activation=activation, up_fct=self.up_fct, n_extra_resnet_layers=self.n_extra_resnet_layers, no_skip_connection=self.no_skip_connection, no_down_up=self.no_down_up),
                Up_G(n_channels * 8, up_out_channels[1], activation=activation, up_fct=self.up_fct, n_extra_resnet_layers=self.n_extra_resnet_layers, no_skip_connection=self.no_skip_connection, no_down_up=self.no_down_up),
                Up_G(n_channels * 4, up_out_channels[2], activation=activation, up_fct=self.up_fct, n_extra_resnet_layers=self.n_extra_resnet_layers, no_skip_connection=self.no_skip_connection, no_down_up=self.no_down_up),
                Up_G(n_channels * 2, up_out_channels[3], activation=activation, up_fct=self.up_fct, n_extra_resnet_layers=self.n_extra_resnet_layers, no_skip_connection=self.no_skip_connection, no_down_up=self.no_down_up),
            ]
        )

        for j, down_in_ch in enumerate(down_in_channels): 
            if self.multi_res_loss or self.sequ_mode or j == 0: 
                self.image_proj_list.append(PartialResnetConvBlock(insize, down_in_ch, activation=activation))
            else: 
                self.image_proj_list.append(nn.Identity())

        self.n_levels = len(self.down)
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        # tail
        self.final_list = nn.ModuleList([])
        for j, up_out_ch in enumerate(up_out_channels): 
            if self.multi_res_loss or self.sequ_mode or j == self.n_levels - 1: 
                self.final_list.append(nn.Conv2d(up_out_ch, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            else: 
                self.final_list.append(nn.Identity())


    def forward(self, x, n_levels_used = None):
        if n_levels_used is None:
            n_levels_used = self.n_levels
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        # print("head, x: ", x.shape)
        h = self.image_proj_list[self.n_levels - n_levels_used](x)

        skip = []
        skip.append(h)
        if self.multi_res_loss: 
            x_list = []
        for i in list(range(self.n_levels))[-n_levels_used:]:
            # finest_level = i == n_levels_used - 1
            # print("down, i: ", i, "h: ", h.shape)
            h = self.down[i](h)  # , finest_level=finest_level
            if i != self.n_levels - 1:
                skip.append(h)

        for j in list(range(n_levels_used)): 
            s = skip.pop()
            # finest_level = j == n_levels_used - 1
            # print("up, j: ", j, "h: ", h.shape, "s: ", s.shape)
            h = self.up[j](h, s)  # , finest_level=finest_level
            if self.multi_res_loss: 
                out = self.final_list[j](h)
                # reshape correctly 
                out = out.reshape(out.shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *out.shape[2:])
                x_list.append(out)
        
        if self.multi_res_loss: 
            return x_list
        else: 
            # print("tail, h: ", h.shape)
            x = self.final_list[n_levels_used - 1](h)
            x = x.reshape(orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:])
            return x

    # def forward(self, x, n_levels_used):
    #     assert x.dim() == 5
    #     orig_shape = x.shape
    #     x = x.reshape(x.size(0), -1, *x.shape[3:])
    #     h = self.image_proj_list[0](x)

    #     x1 = self.down[0](h)
    #     x2 = self.down[1](x1)
    #     x3 = self.down[2](x2)
    #     x4 = self.down[3](x3)
    #     x = self.up[0](x4, x3)
    #     x = self.up[1](x, x2)
    #     x = self.up[2](x, x1)
    #     x = self.up[3](x, h)

    #     x = self.final_list[3](x)
    #     return x.reshape(
    #         orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
    #     )
