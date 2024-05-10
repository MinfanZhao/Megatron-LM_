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

CLASS_TOKEN_LENGTH = 8

class SwinMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes, norm_layer=torch.nn.LayerNorm):
        super(SwinMlpHead, self).__init__()
        self.head = torch.nn.Linear(hidden_size, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        x = self.head(x)
        return x

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

class SwinTransformerBackbone(MegatronModule):
    """Backbone of Swin Transformer Model"""

    def __init__(self, pre_process=True, post_process=True, class_token=False, single_token_output=True):
        super(SwinTransformerBackbone, self).__init__(share_word_embeddings=False)
        args=get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
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
        self.class_token = class_token
        self.hidden_size = args.hidden_size
        
        
        self.patch_size = args.patch_size 
        self.img_size = args.img_size
        assert self.img_size % self.patch_size == 0
        self.in_chans = args.in_chans
        self.depths = args.depths

        self.ape = args.ape
        self.patch_norm = args.patch_norm


        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output

        self.drop_rate = args.drop_rate
        self.drop_path_rate = args.drop_path_rate

        self.num_patches_per_dim_h = self.img_size // self.patch_size
        self.num_patches_per_dim_w = self.img_size // self.patch_size
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w # N = Ph * Pw
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0) # N += [class_token]

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)

            # patch embedding, we use native PatchEmbed function in Swin Transformer rather than linear encoder in vit
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.hidden_size,
                norm_layer=torch.nn.LayerNorm if self.patch_norm else None)

            # absolute position embedding
            if self.ape:
                self.absolute_pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            self.pos_drop = torch.nn.Dropout(p=self.drop_rate)

            # I'm not sure where this argument would be used, so I just keep it.
            args.class_token_present = self.class_token

        # Transformer
        self.transformer = ParallelSwinTransformer(
            self.init_method,
            self.scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            drop_path_rate=self.drop_path_rate,
        )

        # last layer norm will be done in transformer's post process(final layer norm)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

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
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, x):

        if self.pre_process:
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

        x = self.transformer(x, None)

        # last layer norm will be done in transformer's post process(final layer norm), so we comment the following layernorm operation.
        # x = self.norm(x)  # B L C
        if self.single_token_output:
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
        return x


