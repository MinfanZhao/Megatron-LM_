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

"""CV Transformer model."""

import torch
from megatron import get_args
from megatron.model.utils import get_linear_layer
from megatron.model.vision.wenhai_backbone import WenhaiBackbone
from megatron.model.module import MegatronModule
from torch import nn


class WenhaiModel(MegatronModule):
    """Swin Transformer Model."""

    def __init__(self, pre_process=True, post_process=True):
        super(WenhaiModel, self).__init__()
        args = get_args()
        self.num_stages = len(args.depths)
        hidden_size = args.hidden_size
        self.num_features = int(hidden_size * 2 ** (self.num_stages - 1))
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_word_embeddings = False
        # reserve the name "language model" to use fastmoe easier
        # actually it is a vision model
        self.language_model = WenhaiBackbone(
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.language_model.set_input_tensor(input_tensor[0])

    def forward(self, x, x_bulk):
        x = self.language_model(x, x_bulk)

        return x
