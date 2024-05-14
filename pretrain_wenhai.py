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
    """Build the batch."""
    # # if data_iterator is not None:
    # x = torch.ones(1, 93, 1, 4320).half().cuda()
    # x_bulk = torch.ones(1, 9, 1, 4320).half().cuda()
    # x_next = torch.ones(1, 93, 1, 4320).half().cuda()
    # x = x.repeat(1,1,2041,1)
    # x_bulk = x_bulk.repeat(1,1,2041,1)
    # x_next = x_next.repeat(1,1,2041,1)
    # return x, x_bulk, x_next
    """Generate a batch"""
    args = get_args()

    # Items and their type.
    keys = ['x','x_bulk','x_next']
    datatype = torch.half
    # print("in get_batch")
    # Broadcast data.
    if data_iterator is not None:
        # print("get batch data")
        # print("data_iterator: ", data_iterator)
        x, x_next, x_bulk = next(data_iterator)
        
        # print("do next")
        x = x.half().cuda().repeat(1,1,2041,1)
        x_bulk = x_bulk.half().cuda().repeat(1,1,2041,1)
        x_next = x_next.half().cuda().repeat(1,1,2041,1)
        data = {'x':x_next, 'x_bulk':x_bulk, 'x_next':x_next}
    else:
        # print("no data")
        data = None
    # print("before broadcast")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # print('after broadcast')

    # Unpack.
    x = data_b['x']
    x_bulk = data_b['x_bulk']
    x_next = data_b['x_next']
    
    return x, x_next, x_bulk

def loss_func(labels, output_tensor):
    
    labels = labels
    output_tensor = output_tensor
    logits = output_tensor.contiguous().float()
    loss = F.l1_loss(logits, labels)

    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    x, x_next, x_bulk = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(x, x_bulk)
    # print(f"[pretrain wenhai forward step] output_tensor shape: {output_tensor.shape} {output_tensor.is_contiguous()}") 

    return output_tensor, partial(loss_func, x_next)


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
