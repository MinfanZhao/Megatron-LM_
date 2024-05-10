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
from megatron.model.vision.classification import SwinTransformerClassificationModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_ltor_masks_and_position_ids_without_eod
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Swin-Transformer model ...')
    args = get_args()
    model = SwinTransformerClassificationModel(num_classes=args.num_classes,
                                            pre_process=pre_process,
                                            post_process=post_process)
    return model


def get_batch(data_iterator):
    """Build the batch."""
    
    keys = ['image', 'target']
    datatype = torch.float16

    # Broadcast data. Only tensor-model-parallel is available.
    if data_iterator is not None:
        data = next(data_iterator)
        images = data['image'].cuda()
        labels = data['target'].cuda()
        data['target'] = data['target'].to(torch.float16)
        # print('before broadcast',images.dtype,labels.dtype,images.shape,labels.shape)
        # data[1] = data[1].to(torch.float16)
        
    else:
        data = None
    
    
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    images = data_b['image'].cuda()
    labels = data_b['target'].to(torch.int64).cuda()
    # print('after broadcast',images.dtype,labels.dtype,images.shape,labels.shape, images.is_contiguous(),labels.is_contiguous())

    # print(f"images shape:{images.shape} labels shape:{labels.shape}")
    return images, labels

def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)
    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    images, labels = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)


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
