# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2

class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
    mae_decoder = 3
 
class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
