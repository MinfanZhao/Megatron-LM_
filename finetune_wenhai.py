# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Train wenhai model."""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.wenhai_dataset import build_finetune_train_test_dataset
from megatron.model.vision.wenhai import WenhaiFinetuneModel
from megatron.train_wenhai import finetune_by_epoch
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Swin-Transformer model ...')
    args = get_args()
    model = WenhaiFinetuneModel(pre_process=pre_process, post_process=post_process)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()

    # Items and their type.
    datatype = torch.half
    # Broadcast data.
    if data_iterator is not None:
        x_days, x_bulk, sea_mask, day_index = next(data_iterator)
        
        # print(f"xdays:{x_days[0].shape} x_bulk:{x_bulk.shape}, sea mask: {sea_mask.shape} {sea_mask.dtype}")
        x_days = {'x_days' : torch.stack(x_days)}
        bulk_and_mask = {'x_bulk':x_bulk, 'sea_mask': sea_mask}
        day_index = {'day_index':day_index}
    else:
        x_days = None
        bulk_and_mask = None
        day_index = None
        print("data iterator is none")
    data_b = {'x_days': x_days['x_days'], 'x_bulk':bulk_and_mask['x_bulk'], 
              'sea_mask':bulk_and_mask['sea_mask'], 'day_index': day_index['day_index'],}
    return data_b

def loss_func(label, sea_mask, output_tensor, loss_weight, non_loss_data=False):
    
    sea_mask = sea_mask.cuda()
    label = label * sea_mask 
    if loss_weight is not None:
        output_tensor = output_tensor * loss_weight
        label = label * loss_weight
    
    loss = F.l1_loss(output_tensor, label)

    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"loss": averaged_loss[0]}


def forward_step(data, model, day=0):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    x_days = data['x_days']
    x_bulk = data['x_bulk']
    sea_mask = data['sea_mask']
    day_index = data['day_index'].item()
    x_next = x_days[day + 1].cuda()
    timers('batch-generator').stop()
    output_tensor = model(x_days[0], x_bulk, day_index + day)
    # print(f"[pretrain wenhai forward step] output_tensor shape: {output_tensor.shape} {output_tensor.is_contiguous()}") 

    return output_tensor, partial(loss_func, x_next, sea_mask)


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Wenhai finetune ...')
    train_ds, valid_ds = build_finetune_train_test_dataset(args.finetune_days)
    print_rank_0("> finished creating Wenhai finetune datasets ...")

    return train_ds, valid_ds


if __name__ == "__main__":

    finetune_by_epoch(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             get_batch,
    )
