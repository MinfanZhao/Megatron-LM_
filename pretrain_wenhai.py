# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Swin-Transformer"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vision.wenhai import WenhaiModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_ltor_masks_and_position_ids_without_eod
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F


land_sea_mask = torch.ones(1, 1, 2041, 4320).half().cuda()

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Swin-Transformer model ...')
    args = get_args()
    model = WenhaiModel(pre_process=pre_process, post_process=post_process)
    return model


def get_batch(data_iterator):
    """Build the batch."""
    # if data_iterator is not None:
    x = torch.ones(1, 93, 1, 4320).half().cuda()
    x_bulk = torch.ones(1, 9, 1, 4320).half().cuda()
    x_next = torch.ones(1, 93, 1, 4320).half().cuda()
    x = x.repeat(1,1,2041,1)
    x_bulk = x_bulk.repeat(1,1,2041,1)
    x_next = x_next.repeat(1,1,2041,1)
    return x, x_bulk, x_next

def loss_func(labels, output_tensor):
    
    labels = labels * land_sea_mask
    output_tensor = output_tensor * land_sea_mask
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
    x, x_bulk, x_next = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(x, x_bulk)
    # print(f"[pretrain wenhai forward step] output_tensor shape: {output_tensor.shape} {output_tensor.is_contiguous()}") 

    return output_tensor, partial(loss_func, x_next)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Swin ...')
    train_ds, valid_ds = build_train_valid_datasets(
        data_path=args.data_path,
        image_size=(args.img_size, args.img_size)
    )
    print_rank_0("> finished creating Swin datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'dataloader_type': 'cyclic'}
    )
