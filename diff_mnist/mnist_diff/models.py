
from pytorch_wavelets import DWTForward, DWTInverse
import torch

# from .basic import ScoreNetwork1


from torch_ddpm.ddpm.models.unet.layers import *



class DTWBlock(nn.Module): 
    def __init__(self, J, out_channels, mode='zero', wave='haar') -> None:
        super().__init__()

        # choose by user (see below)
        self.version = 1

        # Version 2
        self.J = J
        if self.version == 2 and J == 0: 
            self.xfm = DWTForward(J=J+1, mode=mode, wave=wave)  # Attention! special case.
        else: 
            # Version 1 or j > 0
            self.xfm = DWTForward(J=J, mode=mode, wave=wave)
            self.ifm = DWTInverse(mode=mode, wave=wave)
        self.out_channels = out_channels

    def forward(self, x):
        if self.J == 0: 
            # return identity, but with correct number of channels
            # Version 1: pass image itself on J == 0 
            if self.version == 1: 
                out = x.repeat(1, int(self.out_channels / x.shape[1]) + 1, 1, 1)[:, :self.out_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
            # Version 2: pass HL, LL, LH with zero padding on J == 0
            if self.version == 2: 
                Yl, Yh = self.xfm(x)
                out = torch.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3])).to(x.device)
                # a third of the channels are HL, LL and LH with zero-padding repeated, respectively
                res = Yl.shape[2]
                # a) 
                out[:, 0:int(self.out_channels/3), :res, :res] = Yh[0][:, :, 0, :, :]
                out[:, int(self.out_channels/3):2*int(self.out_channels/3), :res, :res] = Yh[0][:, :, 1, :, :]
                out[:, 2*int(self.out_channels/3):, :res, :res] = Yh[0][:, :, 2, :, :]
                # b) 
                # out[:, 0:int(self.out_channels/4), :res, :res] = Yh[0][:, :, 0, :, :]
                # out[:, int(self.out_channels/4):2*int(self.out_channels/4), :res, :res] = Yh[0][:, :, 1, :, :]
                # out[:, 2*int(self.out_channels/4):3*int(self.out_channels/4), :res, :res] = Yh[0][:, :, 2, :, :]
                # out[:, 3*int(self.out_channels/4):, :res, :res] = Yl
                # c)
                # h_1 = Yh[0][:, :, 0, :, :]
                # h_2 = Yh[0][:, :, 1, :, :]
                # h_3 = Yh[0][:, :, 2, :, :]
                # n_repeat = int(self.out_channels/3)
                # h_1 = h_1.repeat(1, n_repeat, 1, 1)
                # h_2 = h_2.repeat(1, n_repeat, 1, 1)
                # h_3 = h_3.repeat(1, int(self.out_channels - 2*n_repeat), 1, 1)
                # out[:, 0:n_repeat, :res, :res] = h_1
                # out[:, n_repeat:2*n_repeat, :res, :res] = h_2
                # out[:, 2*n_repeat:, :res, :res] = h_3


            # print("out shape")
            # print(out.shape)
            # print("stop")


        else: 
            Yl, Yh = self.xfm(x)
            Yl_1_inv = self.ifm((Yl, []))   
            out = Yl_1_inv

            # before the above lines of downsampling, x is in a certain range, and DWTForward changes the scale by a factor 2^J
            # hence normalize batch_x back to the original range
            out = out / math.pow(2, self.J)  # correct scaling to ensure we are in the original data range

            # copy out across channel dim s.t. number of out channels is out_channels
            out = out.repeat(1, int(self.out_channels / out.shape[1]) + 1, 1, 1)[:, :self.out_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
            
            assert out.shape[1] == self.out_channels

        return out

def get_unet_wavelet_enc(image_size, image_channels, num_channels=32, dropout=0.0, num_res_blocks = 2):
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "168"
    use_checkpoint = False
    use_scale_shift_norm = True

    # Comment: Channel multiplier governs the factor by which the
    # number of channels in the U-Net are multiplied, and also the number
    # of layers the U-Net has.
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2)
    # added
    elif image_size == 16:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 8:
        channel_mult = (1, 2, 2)
    elif image_size in [4, 2]:
        channel_mult = (1, 2)
    elif image_size == 1: 
        channel_mult = (1,)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    kwargs = {
        "in_channels": image_channels,
        "model_channels": num_channels,
        "out_channels": image_channels,
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": tuple(attention_ds),
        "dropout": dropout,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": use_checkpoint,
        "num_heads": num_heads,
        "num_heads_upsample": num_heads_upsample,
        "use_scale_shift_norm": use_scale_shift_norm,
    }
    return UNetWaveletEncModel(**kwargs)



# class UNetWaveletEncModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     """

#     def __init__(
#         self,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         num_classes=None,
#         use_checkpoint=False,
#         num_heads=1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#     ):
#         super().__init__()

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads
#         self.locals = [
#             in_channels,
#             model_channels,
#             out_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout,
#             channel_mult,
#             conv_resample,
#             dims,
#             num_classes,
#             use_checkpoint,
#             num_heads,
#             num_heads_upsample,
#             use_scale_shift_norm,
#         ]
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.num_classes = num_classes
#         self.use_checkpoint = use_checkpoint
#         self.num_heads = num_heads
#         self.num_heads_upsample = num_heads_upsample

#         time_embed_dim = model_channels * 4
#         self.time_embed_list = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        
#         # this block changes the number of channels of the input
#         # self.input_blocks = nn.ModuleList(
#         #     [
#         #         TimestepEmbedSequential(
#         #             conv_nd(dims, in_channels, model_channels, 3, padding=1)
#         #         )
#         #     ]
#         # )
#         # input_block_chans = [model_channels]

#         ch = model_channels
#         ds = 1

#         self.input_blocks = nn.ModuleList([TimestepEmbedSequential(DTWBlock(J=0, out_channels=ch))])  # just feeding the image
#         input_block_chans = [model_channels]


        
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 # print("level: ", level, "mult: ", mult)
                
#                 # DWT does not change number of input channels
#                 ch = int(mult * model_channels)  # GroupNorm32 requires channels to be multiple of 32 (ensured by model_channels input)

#                 # DWT
#                 # done multiple times here (less efficient, could instead just do once per level)
#                 # J=0 on first level: no downsampling -> just feed input, but with right number of channels
#                 # TODO vary wavelet type, but requires fixing of padding
#                 layers = [DTWBlock(J=level, out_channels=ch)]
        
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 input_block_chans.append(ch)

#             # replacing the Downsample block with a DWT block on a lower resolution
#             if level != len(channel_mult) - 1: 

#                 # taking the next level's number of chnanels
#                 ch_downsample = int(channel_mult[level+1] * model_channels)
#                 self.input_blocks.append(TimestepEmbedSequential(
#                     DTWBlock(J=level+1, out_channels=ch_downsample))  # the second +1 is because this DTW block is replacing the Downsample block --> already on lower resolution
#                 )
#                 input_block_chans.append(ch_downsample)
#                 ds *= 2

            

#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )

#         self.output_blocks = nn.ModuleList([])
#         # this reverses the list
#         for level, mult in list(enumerate(channel_mult))[::-1]:
#             for i in range(num_res_blocks + 1):
#                 layers = [
#                     ResBlock(
#                         ch + input_block_chans.pop(),
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels * mult,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = model_channels * mult
#                 if ds in attention_resolutions:
#                     layers.append(
#                         AttentionBlock(
#                             ch,
#                             use_checkpoint=use_checkpoint,
#                             num_heads=num_heads_upsample,
#                         )
#                     )
#                 if level and i == num_res_blocks:  # taken out: "level and " --> in all levels (in contrast to encoder) added, since first manual block in input_blocks not added here (pendant to that)
#                     # print("upsample", level)
#                     layers.append(Upsample(ch, conv_resample, dims=dims))
#                     ds //= 2
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))

#         self.out = nn.Sequential(
#             normalization(ch),
#             SiLU(),
#             zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
#         )

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         self.output_blocks.apply(convert_module_to_f32)

#     def forward(self, x, t, y=None):

#         """
#         Apply the model to an input batch.
#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         timesteps = t.squeeze()
#         assert (y is not None) == (
#             self.num_classes is not None
#         ), "must specify y if and only if the model is class-conditional"

#         hs = []
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)

#         h = x  # .type(self.inner_dtype)
#         for i, module in enumerate(self.input_blocks):
#             h = module(x, emb)
#             # print("in: ", i, h.shape[2], h.shape[1])
#             hs.append(h)
#         h = self.middle_block(h, emb)
#         for i, module in enumerate(self.output_blocks):
            
#             h_cat = hs.pop()
#             # h_cat = torch.zeros_like(h_cat)  # TODO NO SKIP CONNECTIONS; ONLY FOR TESTING

#             # ONLY FOR TESTING: WITHOUT SKIP CONNECTIONS
#             # SHOULD USUALLY BE COMMMENTED OUT!!!!
#             # h_cat = torch.zeros_like(h_cat)  

#             # print("out: ", i, h.shape[2], h_cat.shape[2], h.shape[1], h_cat.shape[1])
#             cat_in = th.cat([h, h_cat], dim=1)  # Comment: skip connections
#             # print("out2: ", cat_in.shape, emb.shape)
#             h = module(cat_in, emb)
#         h = h.type(x.dtype)
        
#         return self.out(h)


    # TODO not used right now / ever?
    # def get_feature_vectors(self, x, timesteps, y=None):
    #     """
    #     Apply the model and return all of the intermediate tensors.
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: a dict with the following keys:
    #              - 'down': a list of hidden state tensors from downsampling.
    #              - 'middle': the tensor of the output of the lowest-resolution
    #                          block in the model.
    #              - 'up': a list of hidden state tensors from upsampling.
    #     """
    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)
    #     result = dict(down=[], up=[])
    #     h = x  # .type(self.inner_dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #         result["down"].append(h.type(x.dtype))
    #     h = self.middle_block(h, emb)
    #     result["middle"] = h.type(x.dtype)
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=1)
    #         h = module(cat_in, emb)
    #         result["up"].append(h.type(x.dtype))
    #     return result