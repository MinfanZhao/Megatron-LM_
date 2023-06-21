# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

from random import shuffle
import torch
from torch.utils.data import Subset, DataLoader
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import LLaMAModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

from tasks.finetune_utils import finetune
from tasks.chat.data import FakeDataSet, KXDigitDataset, build_attn_mask_and_position_ids_with_padding


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building LLaMAModel model ...')
    model = LLaMAModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(batch):
    """Generate a batch"""
    if not isinstance(batch, dict):
        batch = next(batch)
    # Items and their type.
    keys = list(batch.keys())
    print(type(batch), keys)
    
    datatype = torch.int64

    data_b = tensor_parallel.broadcast_data(keys, batch, datatype)

    # Unpack.
    tokens = data_b['input_ids'][:,:-1].long()
    labels = data_b['labels'][:,1:].long()    

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = build_attn_mask_and_position_ids_with_padding(data_b['attention_mask'][:,:-1], tokens.device)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for SFT ...')
    
    train_ds = KXDigitDataset(args.test_data_path[0])
    valid_ds = KXDigitDataset(args.test_data_path[0])
    
    print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds


def main():
    
    finetune(train_valid_datasets_provider, model_provider, model_type=ModelType.encoder_or_decoder, forward_step=forward_step)


if __name__ == "__main__":

    pretrain(train_valid_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'LLaMASentencePieceTokenizer'}
    )
