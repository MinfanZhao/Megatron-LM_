# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Train wenhai model."""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.wenhai_dataset import build_pretrain_train_test_dataset
from megatron.model.vision.wenhai import WenhaiModel
from megatron.train_wenhai import pretrain_by_epoch
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Swin-Transformer model ...')
    args = get_args()
    model = WenhaiModel(pre_process=pre_process, post_process=post_process)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    
    def data_broad_cast_helper(data, datatype):
        return tensor_parallel.broadcast_data({'x'}, {'x':data}, datatype)['x']

    # Items and their type.
    keys = ['x','x_bulk','x_next', 'sea_mask']
    datatype = torch.half
    # Broadcast data.
    if data_iterator is not None:
        x, x_next, x_bulk, sea_mask, loss_weight = next(data_iterator)
        
        # print(f"before broadcast x:{x}")
    else:
        x, x_next, x_bulk, sea_mask, loss_weight = None, None, None, None, None
    x = data_broad_cast_helper(x, torch.half)
    x_next = data_broad_cast_helper(x_next, torch.half)
    x_bulk = data_broad_cast_helper(x_bulk, torch.half)
    sea_mask = data_broad_cast_helper(sea_mask, torch.half)
    loss_weight = data_broad_cast_helper(loss_weight, torch.half)

    # print(f"after broadcast x:{x}")
    
    return x, x_next, x_bulk, sea_mask, loss_weight

def loss_func(labels, sea_mask, loss_weight, output_tensor, non_loss_data=False):
    
    labels = labels * sea_mask
    loss = F.l1_loss(output_tensor * loss_weight, labels * loss_weight)

    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    x, x_next, x_bulk, sea_mask, loss_weight = get_batch(data_iterator)
    x = x.cuda()
    x_next = x_next.cuda()
    x_bulk = x_bulk.cuda()
    sea_mask = sea_mask.cuda()
    loss_weight = loss_weight.cuda()
    timers('batch-generator').stop()
    
    output_tensor = model(x, x_bulk)
   
    return output_tensor, partial(loss_func, x_next, sea_mask, loss_weight)


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Wenhai pretrain ...')
    train_ds, valid_ds = build_pretrain_train_test_dataset()
    print_rank_0("> finished creating Wenhai pretrain datasets ...")

    return train_ds, valid_ds


if __name__ == "__main__":

    pretrain_by_epoch(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
    )
