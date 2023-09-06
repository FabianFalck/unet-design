import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch, type='conv'):
        super().__init__()
        if type == 'conv':
            self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
            self.initialize()
        elif type == 'avg_pool': 
            self.main = nn.AvgPool2d(2)
        else: 
            raise NotImplementedError

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()

        # for debugging purposes
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


# Note: We only use the merged class below going forward 
# class UNet(nn.Module):
#     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
#         super().__init__()
#         assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
#         tdim = ch * 4
#         self.time_embedding = TimeEmbedding(T, ch, tdim)

#         self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
#         self.downblocks = nn.ModuleList()
#         chs = [ch]  # record output channel when dowmsample for upsample
#         now_ch = ch
#         for i, mult in enumerate(ch_mult):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks):
#                 self.downblocks.append(ResBlock(
#                     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 now_ch = out_ch
#                 chs.append(now_ch)
#             if i != len(ch_mult) - 1:
#                 self.downblocks.append(DownSample(now_ch))
#                 chs.append(now_ch)

#         self.middleblocks = nn.ModuleList([
#             ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
#             ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
#         ])

#         self.upblocks = nn.ModuleList()
#         for i, mult in reversed(list(enumerate(ch_mult))):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks + 1):
#                 self.upblocks.append(ResBlock(
#                     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 now_ch = out_ch
#             if i != 0:
#                 self.upblocks.append(UpSample(now_ch))
#         assert len(chs) == 0

#         self.tail = nn.Sequential(
#             nn.GroupNorm(32, now_ch),
#             Swish(),
#             nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
#         )
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.head.weight)
#         init.zeros_(self.head.bias)
#         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
#         init.zeros_(self.tail[-1].bias)

#     def forward(self, x, t):
#         # Timestep embedding
#         temb = self.time_embedding(t)
#         # Downsampling
#         h = self.head(x)
#         hs = [h]
#         for layer in self.downblocks:
#             h = layer(h, temb)
#             hs.append(h)
#         # Middle
#         for layer in self.middleblocks:
#             h = layer(h, temb)
#         # Upsampling
#         for layer in self.upblocks:
#             if isinstance(layer, ResBlock):
#                 h = torch.cat([h, hs.pop()], dim=1)
#             h = layer(h, temb)
#         h = self.tail(h)

#         assert len(hs) == 0
#         return h



# ---------------------------------


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


class UNetWaveletEnc(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, dwt_encoder=False, multi_res_loss=False, downsample_type='conv'):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4

        self.n_levels = len(ch_mult)
        self.dwt_encoder = dwt_encoder
        self.multi_res_loss = multi_res_loss
        self.downsample_type = downsample_type

        self.time_embedding_list = nn.ModuleList(TimeEmbedding(T, ch, tdim) for _ in range(self.n_levels))

        # self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.head_list = nn.ModuleList([])

        self.downblocks = nn.ModuleList([nn.ModuleList() for _ in range(self.n_levels)])
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        # print("ch_mult", ch_mult)
        for l, mult in enumerate(ch_mult):
            self.head_list.append(DTWBlock(J=0, out_channels=now_ch))
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                # self.downblocks.append(ResBlock(
                #     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                #     dropout=dropout, attn=(i in attn)))
                # print("i, out_ch", i, out_ch)
                if self.dwt_encoder: 
                    self.downblocks[l].append(DTWBlock(J=0, out_channels=out_ch))
                else: 
                    self.downblocks[l].append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(l in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if l != len(ch_mult) - 1:
                # self.downblocks.append(DownSample(now_ch))
                # ch_downsample = ch_mult[i+1] * ch
                # print("down: i, out_ch", i, ch_downsample)
                if self.dwt_encoder: 
                    self.downblocks[l].append(DTWBlock(J=1, out_channels=now_ch))
                else: 
                    self.downblocks[l].append(DownSample(now_ch, type=self.downsample_type))
                chs.append(now_ch)
            
        # print("chs encoder", chs)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList([nn.ModuleList() for _ in range(self.n_levels)])
        for l, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for j in range(num_res_blocks + 1):  # TODO why +1? 
                chs_pop = chs.pop()
                # print("up resblocks: l, chs_pop, now_ch",  l, chs_pop, now_ch)
                self.upblocks[l].append(ResBlock(
                    in_ch=chs_pop + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(l in attn)))
                now_ch = out_ch
            if l != 0:   # with indend:  and j == num_res_blocks
                self.upblocks[l].append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail_list = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(32, ch * mult),
            Swish(),
            nn.Conv2d(ch * mult, 3, 3, stride=1, padding=1)
        ) for mult in ch_mult])  

        # for l, mult in enumerate(ch_mult):
        #     out_ch = ch * mult
        
        self.initialize()

    def initialize(self):
        # init.xavier_uniform_(self.head.weight)
        # init.zeros_(self.head.bias)
        for tail in self.tail_list: 
            # TODO should we keep this initialization?
            init.xavier_uniform_(tail[-1].weight, gain=1e-5)
            init.zeros_(tail[-1].bias)

    def forward(self, x, t, n_levels_used=-1):
        if n_levels_used == -1:
            n_levels_used = self.n_levels 

        # Downsampling
        h = self.head_list[-n_levels_used](x)
        hs = [h]
        for level in list(range(self.n_levels))[-n_levels_used:]:
            # choose the appropriate timestep embedding
            temb = self.time_embedding_list[level](t)
            layer_list = self.downblocks[level]
            for layer in layer_list:
                # print("fwd, Enc . : level {}. layer {}".format(level, layer.__class__.__name__))
                if self.dwt_encoder: 
                    h = layer(h)
                else: 
                    h = layer(h, temb)
                
                # print("enc: l, enc", level, h.shape)
                hs.append(h)
        # Middle
        for layer in self.middleblocks:
            # choose the appropriate timestep embedding
            temb = self.time_embedding_list[self.n_levels - 1](t)  # middle blocks belong to last level
            # print("fwd, Mid % : layer {}".format(layer.__class__.__name__))
            h = layer(h, temb)


        model_out_list = []

        # Upsampling
        # self.upblocks has index which is levels, hence reversing
        for l, layer_list in enumerate(self.upblocks[::-1][:n_levels_used]):
            # reversed(list(enumerate(...))) work here as above, because we are at the same time slicing the list, and we want to still have l to be the level 'index'. hence creating this index manually.
            l = len(self.upblocks) - l - 1   # level
            for layer in layer_list:   # do NOT reverse here again.
                if isinstance(layer, ResBlock):
                    # choose the appropriate timestep embedding
                    temb = self.time_embedding_list[l](t)
                    # print("fwd, Dec # : level {}. layer {}".format(l, layer.__class__.__name__))
                    h_cat = hs.pop()
                    # print("dec: h, h_cat, l", h.shape, h_cat.shape, l)
                    h = torch.cat([h, h_cat], dim=1)
                    h = layer(h, temb)
                elif isinstance(layer, UpSample):   # upsample case 
                    # choose the appropriate timestep embedding
                    # note the l-1: this is due to the asymmetry of the encoder and decoder: upsample is on 'lower' layer, downsample is on 'higher' layer
                    # note: on l==0, there is no UpSample block, i.e. not running out of bounds
                    temb = self.time_embedding_list[l-1](t)  
                    last_level_currently_used = self.n_levels - n_levels_used  
                    
                    # first, get output for loss
                    if self.multi_res_loss and l != last_level_currently_used:   # second condition required since on finest level it is called below
                        # print("dec bef tail: l, h", l, h.shape)
                        # print("fwd, Dec # : level {}. Tail.".format(l))
                        out = self.tail_list[l](h)
                        # print("l, out", l, out.shape)
                        model_out_list.append(out)

                    # do upsampling, if not on last level currently used
                    if l != last_level_currently_used:
                        # print("fwd, Dec # : level {}. layer {}".format(l, layer.__class__.__name__))
                        # print("do upsampling!")
                        h = layer(h, temb)
        
        # this has no condition as it is required in all cases: 
        # 1) on the very finest resolution, there is no UpSample layer, i.e. the above out computation doesn't happen, but we still need to compute out
        # 2) if self.multi_res_loss is False, we need to compute out here
        # this case applies both with and without the multi-res loss
        # print("dec bef tail: h", h.shape)
        # print("fwd, Dec # : level {}. Tail.".format(self.n_levels - n_levels_used))
        out = self.tail_list[self.n_levels - n_levels_used](h)  # index 'strange' because index corresponds to level.
        # print("out separate", out.shape)
        model_out_list.append(out)

        assert len(hs) == 0

        # print("fwd, pass end ---")

        if self.multi_res_loss:
            assert len(model_out_list) == n_levels_used
            return model_out_list
        else: 
            assert len(model_out_list) == 1
            return model_out_list[-1]


    # ----

    



if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
