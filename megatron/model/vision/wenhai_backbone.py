# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Swin Transformer model."""

import math
import einops
import torch
import apex

from torch import nn
import torch.nn.functional as F
from megatron import get_args
from megatron.model import LayerNorm
from megatron.model.swin_transformer import ParallelSwinTransformer
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from megatron.model.module import MegatronModule

from timm.models.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from megatron.model.vision.calcu_bulk_flux import calc_bulk_flux

class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut
    

class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut
    
    
def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back

def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]

class PatchEmbed(torch.nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class WenhaiBackbone(MegatronModule):
    """Backbone of Swin Transformer Model"""

    def __init__(self, pre_process=True, post_process=True):
        super(WenhaiBackbone, self).__init__(share_word_embeddings=False)
        args=get_args()
        
        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, sum(self.depths)  # args.num_layers = sum(self.depths)
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        
        
        self.patch_size = args.patch_size 
        self.image_size = args.img_size
        self.in_chans = args.in_chans
        self.depths = args.depths
        self.window_size = args.window_size
        self.in_chans = args.in_channels
        self.in_bulk_chans = args.in_bulk_channels
        self.out_chans = args.out_channels

        self.ape = args.ape
        self.patch_norm = args.patch_norm
        self.micro_batch_size = args.micro_batch_size

        self.drop_rate = args.drop_rate
        self.drop_path_rate = args.drop_path_rate
        self.input_tensor = None
        
        padding_size = 2 * self.patch_size[0] * self.window_size
        tp_padding_size = to_2tuple(padding_size)#直接一次padding，满足patch_size和window_size以及下采样需求
        self.initial_padding = get_pad2d(self.image_size, tp_padding_size)
        self.initial_pad = nn.ZeroPad2d(self.initial_padding)
        padding_left, padding_right, padding_top, padding_bottom = self.initial_padding


        self.padded_size = [self.image_size[0] + padding_top + padding_bottom, 
                            self.image_size[1] + padding_left + padding_right]
        #input_resolution = list(input_resolution)
        patches_resolution = [self.padded_size[0] // self.patch_size[0], 
                              self.padded_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        

        

        if self.pre_process:
            self.proj = torch.nn.Conv2d(self.in_chans, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.proj_bulk = torch.nn.Conv2d(self.in_bulk_chans, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.down_blk = DownBlock(self.hidden_size, self.hidden_size, num_groups=32)
            self.down_blk2 = DownBlock(self.hidden_size, self.hidden_size, num_groups=32)
        self.div_val=self.patch_size[0] * 2

        # Transformer
        self.transformer = ParallelSwinTransformer(
            self.init_method,
            self.scaled_init_method,
            post_layer_norm = False,
            pre_process=self.pre_process,
            post_process=self.post_process,
            drop_path_rate=self.drop_path_rate,
            input_resolution=(patches_resolution[0] // 2, patches_resolution[1] // 2),
        )

        if self.post_process:
            self.sea_mask = torch.load('/test2/cuiyz/data/mask_GLORYS_0.083d.pt').unsqueeze(0).half().cuda()
            self.up_blk = UpBlock(self.hidden_size, self.hidden_size, num_groups=32)
            self.fc = nn.Linear(self.hidden_size, self.out_chans * self.patch_size[0] * self.patch_size[1])
            self.previous_result = None # register previous result in last stage, because wenhai only predict the difference between days 
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)


    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        
        if not self.pre_process:
            self.transformer.set_input_tensor(input_tensor)
        else:
            if input_tensor is not None:
                self.input_tensor = input_tensor
                
    
    def clean_finetune_buffer(self):
        self.input_tensor = None
        self.previous_result = None
    

    def forward(self, x, bulk_flux, bulk_flux_idx=0):

        def pre_process_func(x, bulk_flux):
            x = self.initial_pad(x)
            bulk_flux = self.initial_pad(bulk_flux)
            B, C, pad_lat, pad_lon = x.shape
            x = self.proj(x)
            x = self.down_blk(x)
            x = x.flatten(2).transpose(1, 2)
            bulk_flux = self.proj_bulk(bulk_flux)
            bulk_flux = self.down_blk2(bulk_flux)
            bulk_flux = bulk_flux.flatten(2).transpose(1,2)
            x = x + bulk_flux
            return B, C, pad_lat, pad_lon, x
        
        def post_process_func(pad_lat, pad_lon, x):
            B = x.shape[0]
            x = x.view(B, pad_lat//self.div_val, pad_lon//self.div_val, -1)
            x = x.permute(0, 3, 1, 2)#[1, 96, 259, 546]

            x = self.up_blk(x)#[1, 96, 518, 1092]

            x = x.permute(0, 2, 3, 1)

            x = self.fc(x)#[1, 518, 1092, 1488]

            x = rearrange(
                x,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                h=pad_lat // self.patch_size[0],
                w=pad_lon // self.patch_size[1],
            )

            #x shape [1, 93, 2072, 4368]

            padding_left, padding_right, padding_top, padding_bottom = self.initial_padding
            x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]#[1, 93, 2041, 4320]

            return x


        if self.post_process:    
            if self.previous_result is None:
                # set for first day
                self.previous_result = x.cuda()

        if self.pre_process:
            if self.input_tensor is not None:
                x = self.input_tensor
                bulk_flux = calc_bulk_flux(x.detach().cpu(), bulk_flux_idx).cuda()
            else:
                x, bulk_flux = x.cuda(), bulk_flux.cuda()
            # B, C, pad_lat, pad_lon, x = checkpoint(pre_process_func, x, bulk_flux, use_reentrant=False)
            B, C, pad_lat, pad_lon, x = pre_process_func(x, bulk_flux)
            # print(f"x after preprocess:{x.shape} {x}")
               
        if not x.is_contiguous():
            x = x.contiguous()
        
        
            
        x = self.transformer(x, None)
        
        if self.post_process:
            x = post_process_func(self.padded_size[0], self.padded_size[1], x)
            if self.previous_result is not None:
                x = x + self.previous_result
            x = x * self.sea_mask
            self.previous_result = x            
        return x
