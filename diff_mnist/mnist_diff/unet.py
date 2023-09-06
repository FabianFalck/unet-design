
import torch
from torch_ddpm.ddpm.models.unet.layers import *
from mnist_diff.models import DTWBlock
from mnist_diff.layers import UpInterpolate
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTInverse
from utils import compute_norm


def get_unet_wavelet(image_size, image_channels, num_channels=32, dropout=0.0, num_res_blocks = 2, dwt_encoder=False, multi_res_loss=False, model_out_passed_on=False, avg_pool_down=False):
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
        # channel_mult = (1, 2, 3, 4)  # original
        channel_mult = (2, 2, 2, 2)  # G-Net: all channels same
    elif image_size == 32:
        # channel_mult = (1, 2, 2, 2)  # original
        channel_mult = (2, 2, 2, 2)  # G-Net: all channels same
    elif image_size == 28:
        channel_mult = (1, 2, 2)
    # added
    elif image_size == 16:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 8:
        channel_mult = (1, 2, 2)
    elif image_size == 4:
        # channel_mult = (1, 2)  # original
        channel_mult = (1, 1, 1)  # G-Net: all channels same
    elif image_size == 2: 
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
        "dwt_encoder": dwt_encoder,
        "multi_res_loss": multi_res_loss,
        'model_out_passed_on': model_out_passed_on,
        'conv_resample': not avg_pool_down,
    }
    return UNet_wavelet(**kwargs)






class UNet_wavelet(nn.Module):
    """
    The full UNet_wavelet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dwt_encoder=False,
        multi_res_loss=False,
        model_out_passed_on=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.locals = [
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            num_heads,
            num_heads_upsample,
            use_scale_shift_norm,
        ]
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.n_levels = len(channel_mult)
        self.dwt_encoder = dwt_encoder
        self.multi_res_loss = multi_res_loss
        self.model_out_passed_on = model_out_passed_on

        time_embed_dim = model_channels * 4
        self.time_embed_list = nn.ModuleList([nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        ) for _ in range(self.n_levels)])

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        
        # this block changes the number of channels of the input
        # self.input_blocks = nn.ModuleList(
        #     [
        #         TimestepEmbedSequential(
        #             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        #         )
        #     ]
        # )
        # input_block_chans = [model_channels]

        ch = model_channels * channel_mult[0]   # use here first multiplier
        ds = 1

        # TODO !!!!! the first layer could be a resolution-specific (!) ResNet block as well as in unet.py, 
        # TODO currently uses input image in any case, just adjusting to have correct number of channels
        # TODO this would then symmetrically resemble how the self.out_reduce_channels looks like
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(DTWBlock(J=0, out_channels=ch))])  # just feeding the image
        input_block_chans = [ch]

        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if self.dwt_encoder: 
                    # print("level: ", level, "mult: ", mult)
                    
                    # DWT does not change number of input channels
                    ch = int(mult * model_channels)  # GroupNorm32 requires channels to be multiple of 32 (ensured by model_channels input)

                    # DWT
                    # done multiple times here (less efficient, could instead just do once per level)
                    # J=0 on first level: no downsampling -> just feed input, but with right number of channels
                    # TODO vary wavelet type, but requires fixing of padding

                    layers = [DTWBlock(J=0, out_channels=ch)]
            
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                    input_block_chans.append(ch)
                else: 
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                            )
                        )
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                    input_block_chans.append(ch)

            # replacing the Downsample block with a DWT block on a lower resolution
            if level != len(channel_mult) - 1: 
                if self.dwt_encoder: 
                    # taking the next level's number of chnanels
                    ch_downsample = int(channel_mult[level+1] * model_channels)
                    self.input_blocks.append(TimestepEmbedSequential(
                        DTWBlock(J=1, out_channels=ch_downsample))  # the second +1 is because this DTW block is replacing the Downsample block --> already on lower resolution
                    )
                    input_block_chans.append(ch_downsample)
                    ds *= 2
                else: 
                    self.input_blocks.append(
                        TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                    )
                    input_block_chans.append(ch)
                    ds *= 2

            

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.out_f_list = nn.ModuleList([nn.ModuleList() for _ in range(len(channel_mult))])  # list of lists, one for each resolution
        self.out_upsample_list = nn.ModuleList([nn.ModuleList() for _ in range(len(channel_mult))])

        # this reverses the list
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # print("level: ", level, "i: ", i, "out_channels: ", model_channels * mult)
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    print("adding attention")
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                # TimeStepEmbed now done earlier in comparison to the original implementation, not including the Upsample
                self.out_f_list[level].append(TimestepEmbedSequential(*layers))

                if i == num_res_blocks:  # taken out: "level and " --> in all levels (in contrast to encoder) added, since first manual block in input_blocks not added here (pendant to that)
                    if level: 
                        # print("upsample", level)
                        self.out_upsample_list[level].append(TimestepEmbedSequential(*[Upsample(ch, conv_resample, dims=dims)]))
                        ds //= 2
                    else: 
                        self.out_upsample_list[level].append(TimestepEmbedSequential(*[nn.Identity()]))


        # Version 1 G-Net
        # self.out_reduce_channels_yl = nn.Conv2d(in_channels=ch, out_channels=out_channels, kernel_size=1, stride=1)
        # self.out_reduce_channels_yh_list = nn.ModuleList([nn.Conv2d(in_channels=ch, out_channels=out_channels, kernel_size=1, stride=1) for _ in range(len(channel_mult))])
        # self.out_dwt_inv = DWTInverse(mode='zero', wave='haar')

        if self.multi_res_loss: 
            self.out_reduce_channels_list = nn.ModuleList([nn.Conv2d(in_channels=ch, out_channels=out_channels, kernel_size=1, stride=1) for _ in range(len(channel_mult))])
            self.out_activation_list = nn.ModuleList([ nn.Sequential(
                    normalization(ch),
                    SiLU(),
                    # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),   # correponds to reduce channels layer
                    ) for _ in range(len(channel_mult))
                ])
        else: 
            # self.out = nn.Sequential(
            #     normalization(ch),
            #     SiLU(),
            #     # zero_module(conv_nd(dims, model_channels * channel_mult[0], out_channels, 3, padding=1)),
            # )
            # self.out_reduce_channels = nn.Conv2d(in_channels=ch, out_channels=out_channels, kernel_size=1, stride=1)

            self.out_reduce_channels_list = nn.ModuleList([nn.Conv2d(in_channels=ch, out_channels=out_channels, kernel_size=1, stride=1) for _ in range(len(channel_mult))])
            self.out_activation_list = nn.ModuleList([ nn.Sequential(
                    normalization(ch),
                    SiLU(),
                    # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),   # correponds to reduce channels layer
                    ) for _ in range(len(channel_mult))
                ])


        # TODO get rid of this? Is this possible in just the UNet? 
        # self.out = nn.Sequential(
        #     normalization(ch),
        #     SiLU(),
        #     # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        # )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    
    def compute_time_embedding(self, timesteps, level, y=None):
        # special case: in decoder, level==-1 is possible due to upsample
        # in this case, pass the embedding for level==0, which is however in turn not used at all because of the Identity function.
        if level == -1:
            level = 0

        # time step embedding
        emb = self.time_embed_list[level](timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        return emb
    

    def forward(self, x, t, y=None, n_levels_used=-1, u_net_norm=False):

        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        : param n_levels_used: the maximum index of the decoder to be used. -1 indicates using all levels.
        : param superres_n_res: the number of levels added in the decoder during superresolution
        :return: an [N x C x ...] Tensor of outputs.
        """
        if n_levels_used == -1:
            n_levels_used = len(self.channel_mult)
    

        timesteps = t.squeeze()
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        if self.multi_res_loss: 
            hs_f_dec = []

        if u_net_norm: 
            # prepare a dictionary to log norms
            norms = {
                'down': {k: [] for k in range(self.n_levels)},
                'middle': [],
                'up': {k: [] for k in range(self.n_levels)},
            }
        else: 
            norms = None

        # TODO this is now done in the main
        # if superres_n_res > 0: 
        #     # inclusion of input to higher resolution
        #     for s in list(range(superres_n_res))[::-1]:   # in encoder order
        #         for _ in range(self.num_res_blocks + 1): 
        #             mult = self.channel_mult[0]
        #             ch = int(mult * self.model_channels)
        #             increase_channels_layer = TimestepEmbedSequential(DTWBlock(J=0, out_channels=ch))
        #             up_layer = UpInterpolate(up_rate=int(2 ** (s + 1)))
        #             x_included = increase_channels_layer(x, emb)
        #             x_included = up_layer(x_included)
        #             hs.append(x_included)

        h = x  # .type(self.inner_dtype)

        lowest_level = len(self.channel_mult) - n_levels_used
        if u_net_norm: 
            norms['down'][lowest_level].append(compute_norm(h))  
        # in the encoder (with order fine to coarse), use the first n_levels_used levels
        # list NOT sub-divided into levels (as in decoder)
        # to understand indexes, see how self.input_blocks is constructed above

        # - 1 since on last level no downsample, i.e. input_blocks[0] is never used
        upper_range = n_levels_used * (self.num_res_blocks + 1) - 1
        # use input layer (to have the right number of channels), and then the 'lower'/coarser part of the encoder
        ins = nn.ModuleList([self.input_blocks[0]]) + self.input_blocks[::-1][:upper_range][::-1]  # reverse order twice (-> fine to coarse, first reverse only for correct selection)
        
        for i, module in enumerate(ins):   
            # compute the correct level 
            start_level = self.n_levels - n_levels_used
            level = start_level + int((i - 1) / (self.num_res_blocks + 1))  # -1 accounts for the input block at [0]
            # compute timestep embedding
            emb = self.compute_time_embedding(timesteps=timesteps, level=level, y=y)
            
            # print("enc, h bef: ", i, h.shape)
            # print("fwd, Enc . : i {}. level {}. module {}".format(i, level, module[0].__class__.__name__))
            h = module(h, emb)
            if u_net_norm: 
                norms['down'][level].append(compute_norm(h))
            # print("enc, h aft: ", i, h.shape)
            hs.append(h)
        # print("fwd, Mid % : module {}".format(module.__class__.__name__))

        # compute timestep embedding for middle block
        level = self.n_levels - 1  # is on last level (coarsest)
        emb = self.compute_time_embedding(timesteps=timesteps, level=level, y=y)

        h = self.middle_block(h, emb)
        if u_net_norm:
            norms['middle'].append(compute_norm(h))

        self.model_out_passed_on = True  

        if self.multi_res_loss and self.model_out_passed_on: 
            model_out_list = []

        # in the decoder (with order coarse to fine), use the first n_levels_used levels
        level_inv = 0
        for i, level in enumerate(list(range(len(self.channel_mult)))[::-1][:n_levels_used]):  # list sub-divided into levels (as opposed to encoder)
            for out_block in self.out_f_list[level]:
                h_cat = hs.pop()
                # print("dec, i, h, h_cat: ", i, h.shape, h_cat.shape)
                cat_in = th.cat([h, h_cat], dim=1)  # Comment: skip connections
                # print("out2: ", cat_in.shape, emb.shape)
                # print("fwd, Dec # : level {}. module {}".format(level, out_block.__class__.__name__))

                # compute timestep embedding
                emb = self.compute_time_embedding(timesteps=timesteps, level=level, y=y)

                h = out_block(cat_in, emb)  #  Note: what does this do in the case of Upsample? --> see TimestepEmbedSequential
                if u_net_norm:
                    norms['up'][level].append(compute_norm(h))

            # print("before upsample - ", "i: ", i, "h.shape: ", h.shape)
            if self.multi_res_loss:
                if self.model_out_passed_on: 
                    # change state
                    # print("fwd, Dec # : level {}. out_act. module {}.".format(level, self.out_activation_list[i].__class__.__name__))
                    h = self.out_activation_list[i](h)
                    n_state_channels = h.shape[1]
                    # print("fwd, Dec # : level {}. out_rch. module {}.".format(level, self.out_reduce_channels_list[i].__class__.__name__))
                    h = self.out_reduce_channels_list[i](h)  # has to be after activation s.t. output can be outside [-1,1], since is noise
                    # append to outputs
                    model_out_list.append(h)
                    if u_net_norm:
                        norms['up'][level].append(compute_norm(h))
                    # repeat h to get back originally needed channels
                    h = h.repeat(1, int(n_state_channels / h.shape[1]) + 1, 1, 1)[:, :n_state_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
                    if u_net_norm:
                        norms['up'][level].append(compute_norm(h))
                else: 
                    hs_f_dec.append(h)  # store for skip connection within decoder  # TODO USE
            elif self.model_out_passed_on:
                # print("fwd, Dec # : level {}. out_act. module {}.".format(level, self.out_activation_list[i].__class__.__name__))
                h = self.out_activation_list[i](h)
                n_state_channels = h.shape[1]
                # print("fwd, Dec # : level {}. out_rch. module {}.".format(level, self.out_reduce_channels_list[i].__class__.__name__))
                h = self.out_reduce_channels_list[i](h)  # has to be after activation s.t. output can be outside [-1,1], since is noise
                if u_net_norm:
                    norms['up'][level].append(compute_norm(h))
                # last_level_used = self.n_levels - n_levels_used
                if not level_inv == n_levels_used - 1: 
                    h = h.repeat(1, int(n_state_channels / h.shape[1]) + 1, 1, 1)[:, :n_state_channels, :, :]  # +1 and then slicing to cover the case where out_channels is no multiple of input channels
                    if u_net_norm:
                        norms['up'][level].append(compute_norm(h))

            # calls Identity on highest resolution; is not called on level n_levels_used - 1
            if not level_inv == n_levels_used - 1: 
                # compute timestep embedding
                # note the level-1: this is due to the asymmetry of the encoder and decoder: upsample is on 'lower' layer, downsample is on 'higher' layer
                # on level==0, level==-1 is passed to self.compute_time_embedding. As a special case, it returns the embedding for level==0, which is in turn not used at all because of the Identity function.
                emb = self.compute_time_embedding(timesteps=timesteps, level=level-1, y=y)

                # print("fwd, Dec # : level {}. out_ups. module {}.".format(level, self.out_upsample_list[level].__class__.__name__))
                h = self.out_upsample_list[level][0](h, emb)  # only one element is in the list
                if u_net_norm:
                    norms['up'][level].append(compute_norm(h))
            # print("- after upsample - ", "i: ", i, "h.shape: ", h.shape)
            level_inv += 1

        h = h.type(x.dtype)

        # Step 2: reduce the nubmer of channels to input channels
        # and
        # Step 3: pass thorugh sigmoid
        if self.multi_res_loss and self.model_out_passed_on is False: 
            hs_f_dec_new = []
            for i, h in enumerate(hs_f_dec):
                h_new = self.out_activation_list[i](h)
                h_new = self.out_reduce_channels_list[i](h_new)  # has to be after activation s.t. output can be outside [-1,1], since is noise
                hs_f_dec_new.append(h_new)
            hs_f_dec = hs_f_dec_new
            model_out_list = hs_f_dec

        if self.multi_res_loss: 
            return model_out_list, norms
        else: 
            # Version 1: shared output layers
            # h = self.out(h)
            # h = self.out_reduce_channels(h)

            # Version 2: separate output layers
            if not self.model_out_passed_on: 
                # print("fwd, Dec # : level 'final'. out_act. module {}.".format(self.out_activation_list[i].__class__.__name__))
                h = self.out_activation_list[n_levels_used - 1](h)
                # print("fwd, Dec # : level 'final'. out_rch. module {}.".format(self.out_reduce_channels_list[i].__class__.__name__))
                h = self.out_reduce_channels_list[n_levels_used - 1](h)  # has to be after activation s.t. output can be outside [-1,1], since is noise
                if u_net_norm:  # TODO correct? 
                    norms['up'][lowest_level].append(compute_norm(h))

            return h, norms

        

        
        
        # Step SPECIAL: # TODO can we get rid of this? 
        # TODO get rid of this? Is this possible in just the UNet? 
        # TODO for now just commented out
        # Version 1 (see below):
        # out = self.out(h)
        # Version 2: see above


        # Step 3: apply inverse wavelet transform
        # create output format that DWTInverse expects (see https://pytorch-wavelets.readthedocs.io/en/latest/functions.html?highlight=DWTInverse#pytorch_wavelets.DWTInverse)

        # # Version 1: in one step (with out)
        # res = out.shape[2]
        # res_half = int(res / 2)
        # yl = out[:, :, :res_half, :res_half]
        # # order: "LH, HL and HH" (see https://pytorch-wavelets.readthedocs.io/en/latest/functions.html?highlight=DWTInverse#pytorch_wavelets.DWTInverse)
        # # we assume the order   LL HL
        # #                       LH HH    in an image.
        # yh = [out[:, :, :res_half, res_half:], out[:, :, res_half:, :res_half], out[:, :, res_half:, res_half:]]
        # # concatenate tensors in yh list along new dimension 2
        # yh = [a.unsqueeze(2) for a in yh]
        # yh = torch.cat(yh, dim=2)
        # # TODO one could pass an yh list of all the separate levels here, but instead, we just pass it as if there was just one level
        # # TODO does this make a difference? 
        # # TODO test it, instead of constructing out
        # yh = [yh]
        # # expects list
        # out = self.out_dwt_inv((yl, yh))

        # Version 2: using a multi-level DWTInverse
        # out = self.out_dwt_inv((yl_mult, yh_mult_list))

        # return out 





    # ############################################
    # Version 1, finalised 17/2/23
    # ############################################
    # def forward(self, x, t, y=None):

    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     timesteps = t.squeeze()
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"

    #     hs = []
    #     hs_f_dec = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)

    #     h = x  # .type(self.inner_dtype)
    #     for i, module in enumerate(self.input_blocks):
    #         h = module(x, emb)
    #         # print("enc, h: ", i, h.shape)
    #         hs.append(h)
    #     h = self.middle_block(h, emb)

    #     for level in list(range(len(self.channel_mult)))[::-1]: 
    #         for out_block in self.out_f_list[level]:
    #             h_cat = hs.pop()
    #             # print("dec, h, h_cat: ", i, h.shape, h_cat.shape)
    #             cat_in = th.cat([h, h_cat], dim=1)  # Comment: skip connections
    #             # print("out2: ", cat_in.shape, emb.shape)
    #             h = out_block(cat_in, emb)  #  Note: what does this do in the case of Upsample? --> see TimestepEmbedSequential

    #         # print("before upsample - ", "i: ", i, "h.shape: ", h.shape)
    #         hs_f_dec.append(h)  # store for skip connection within decoder  # TODO USE

    #         # calls Identity on highest resolution
    #         h = self.out_upsample_list[level][0](h, emb)  # only one element is in the list
    #         # print("- after upsample - ", "i: ", i, "h.shape: ", h.shape)

    #     h = h.type(x.dtype)

    #     # construct output (one could also do this in the for loop)
    #     # Step 1: construct many-channels output in wavelet domain
    #     # TODO currently limited to all resolutions in decoder having same amount of channels
    #     assert all([h.shape[1] == hs_f_dec[0].shape[1] for h in hs_f_dec])
    #     # output in Step 1 has same dimensions as input image x, but it has many channels
    #     out = torch.zeros(x.shape[0], hs_f_dec[0].shape[1], x.shape[2], x.shape[3]).to(x.device)  
    #     out = out.type(x.dtype)

    #     # --- (DEBUG AND VISUALISE - START) ---
    #     # # debugging purposes: difference indicator matrix (binary)
    #     # diff_ind_full = torch.zeros((x.shape[2], x.shape[3]))
    #     # last_out = out.clone()
    #     # --- (DEBUG AND VISUALISE - END) ---

    #     yl_mult, yh_mult_list = None, []

    #     for i, h in enumerate(hs_f_dec):
    #         res = h.shape[2]
    #         res_half = int(res / 2)

    #         if i == 0: 
    #             # in smallest resolution: take entire output, and put in top left corner (LL)
    #             yl_mult = h
    #             out[:, :, :res, :res] = yl_mult
    #         else: 
    #             # in all other resolutions: take 3/4-th of the output, namely its LH, HL, and HH part (bottom left, top right and bottom right corner)
    #             # Note: pytroch-wavelets expects input as (N, C, H, W)
    #             # top-right corner (HL)
    #             HL = h[:, :, :res_half, res_half:]
    #             out[:, :, :res_half, res_half:res] = HL
    #             # bottom-left corner (LH) 
    #             LH = h[:, :, res_half:, :res_half]
    #             out[:, :, res_half:res, :res_half] = LH
    #             # bottom-right corner (HH)
    #             HH = h[:, :, res_half:, res_half:]
    #             out[:, :, res_half:res, res_half:res] = HH

    #             # zero-out in resnet state what is not used in output
    #             out[:, :, :res_half, :res_half] = 0

    #             # lower index in yh_mult corresponds to finer resolution (see https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html#)
    #             # hence, we append it here in the opposite order, and reverse the list afterwards
    #             yh_mult_list.append(torch.cat((LH.unsqueeze(2), HL.unsqueeze(2), HH.unsqueeze(2)), dim=2))

    #         # --- (DEBUG AND VISUALISE - START) ---
    #         # # difference matrix indicates which values, summed across channels and batches, have changed in out 
    #         # out_diff = torch.sum(out - last_out, dim=(0, 1))
    #         # print(i)
    #         # # print("out: ", torch.sum(out, dim=(0, 1)))
    #         # # print("last_out: ", torch.sum(last_out, dim=(0, 1)))
    #         # # print("out_diff: ", out_diff)
    #         # diff_ind = torch.where(out_diff != 0, torch.ones_like(out_diff) * res, torch.zeros_like(out_diff))  # * res indicates the resolution

    #         # diff_ind_full += diff_ind

    #         # # plot a heatmap of difference indicator matrix on every resolution
    #         # # plt.imshow(diff_ind.cpu().numpy())
    #         # # # add scale
    #         # # plt.colorbar()
    #         # # plt.show()
    #         # # # set breakpoint here
    #         # # plt.close()

    #         # # for next iteration
    #         # last_out = out.clone()
    #         # --- (DEBUG AND VISUALISE - END) ---

    #     # reverse yh_mult, so that lower index corresponds to finer resolution
    #     yh_mult_list = yh_mult_list[::-1]

    #     # --- (DEBUG AND VISUALISE - START) ---
    #     # # plot a heatmap of the full difference indicator matrix  
    #     # diff_ind_full = diff_ind_full.cpu().numpy()
    #     # # plot
    #     # plt.imshow(diff_ind_full)
    #     # # add scale
    #     # plt.colorbar(label="resolution", ticks=np.unique(diff_ind_full).tolist())
    #     # # x ticks from 0 to max resolution
    #     # tick_pos = np.array([0] + [j for j in range(int(res / 4) - 1, res, int(res / 4))] + [res - 1])
    #     # plt.xticks(tick_pos, tick_pos + 1)
    #     # plt.yticks(tick_pos, tick_pos + 1)
    #     # plt.show()
    #     # # set breakpoint here
    #     # plt.close()
    #     # --- (DEBUG AND VISUALISE - END) ---

    #     # Step 2: reduce the nubmer of channels to input channels
    #     # Version 1 (see below): 
    #     # out = self.out_reduce_channels(out)

    #     # Version 2 (see below): 
    #     yl_mult = self.out_reduce_channels_yl(yl_mult)
    #     yh_mult_list_new = []
    #     for i, yh_mult in enumerate(yh_mult_list):
    #         # get out each of 3 'H', and reduce channels
    #         H_list = [yh_mult[:, :, i, :, :] for i in range(yh_mult.shape[2])]

    #         # Step SPECIAL, Version 2 (see below): 
    #         # H_list = [self.out(H) for H in H_list]

    #         # apply convolution
    #         H_list = [self.out_reduce_channels_yh_list[i](H) for H in H_list]
    #         # insert '3' dimension again
    #         H_list = [H.unsqueeze(2) for H in H_list]
    #         # put back together
    #         yh_mult = torch.cat(H_list, dim=2)
    #         # append to result
    #         yh_mult_list_new.append(yh_mult)
    #     yh_mult_list = yh_mult_list_new

        
    #     # Step SPECIAL: # TODO can we get rid of this? 
    #     # TODO get rid of this? Is this possible in just the UNet? 
    #     # TODO for now just commented out
    #     # Version 1 (see below):
    #     # out = self.out(h)
    #     # Version 2: see above


    #     # Step 3: apply inverse wavelet transform
    #     # create output format that DWTInverse expects (see https://pytorch-wavelets.readthedocs.io/en/latest/functions.html?highlight=DWTInverse#pytorch_wavelets.DWTInverse)

    #     # # Version 1: in one step (with out)
    #     # res = out.shape[2]
    #     # res_half = int(res / 2)
    #     # yl = out[:, :, :res_half, :res_half]
    #     # # order: "LH, HL and HH" (see https://pytorch-wavelets.readthedocs.io/en/latest/functions.html?highlight=DWTInverse#pytorch_wavelets.DWTInverse)
    #     # # we assume the order   LL HL
    #     # #                       LH HH    in an image.
    #     # yh = [out[:, :, :res_half, res_half:], out[:, :, res_half:, :res_half], out[:, :, res_half:, res_half:]]
    #     # # concatenate tensors in yh list along new dimension 2
    #     # yh = [a.unsqueeze(2) for a in yh]
    #     # yh = torch.cat(yh, dim=2)
    #     # # TODO one could pass an yh list of all the separate levels here, but instead, we just pass it as if there was just one level
    #     # # TODO does this make a difference? 
    #     # # TODO test it, instead of constructing out
    #     # yh = [yh]
    #     # # expects list
    #     # out = self.out_dwt_inv((yl, yh))

    #     # Version 2: using a multi-level DWTInverse
    #     out = self.out_dwt_inv((yl_mult, yh_mult_list))

    #     return out 