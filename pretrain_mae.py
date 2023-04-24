# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, print_rank_0
from megatron.data.vit_dataset import build_pretrain_datasets
from megatron.model import ModelType
from megatron.model.mae_backbone import MaskedAutoencoderViT
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.core import tensor_parallel
import torchvision.transforms as T

def model_provider(pre_process=True, post_process=True,
                   add_encoder=True, add_decoder=True):
    """Build the model."""

    args = get_args()

    if args.vision_backbone_type == 'mae':
        print_rank_0("building mae model ...")
        model = MaskedAutoencoderViT(
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder
            )
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))
    return model


def get_batch(data_iterator):
    """Build the batch."""

    keys = ['img']
    datatype = torch.half ## not sure 

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    
    images = data_b['img'].half()

    return images, images


def patchify(imgs,patch_dim):
    '''
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    '''
    p = patch_dim
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


def loss_func(labels, output_tensor):
    '''
    output_tensor: output,ids_restore,mask
    output: b s h 
    mask: [b s ] 0 is keep, 1 is remove
    labels: images [b c h w]
    '''
    pred = output_tensor[0].float()
    mask = output_tensor[1]
    args = get_args()
    target = patchify(labels,args.patch_dim) # b s h

    if args.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
    loss = (pred - target.float()) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    averaged_losses = average_losses_across_data_parallel_group([loss])

    return loss, {"loss": averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)
    from megatron.core import mpu
    return output_tensor, partial(loss_func, labels)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building pretrain datasets " "for MAE ..."
    )

    mean = [eval(args.mean.split(',')[i]) for i in range(len(args.mean.split(',')))] if args.mean is not None else [0.485, 0.456, 0.406]
    std = [eval(args.std.split(',')[i]) for i in range(len(args.std.split(',')))] if args.std is not None else [0.229, 0.224, 0.225]
    transform = T.Compose([
            T.RandomResizedCrop((args.img_h, args.img_w), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
    train_ds = build_pretrain_datasets(
        data_path=args.data_path,
        image_size=(args.img_h, args.img_w),
        transform=transform
    )
    print_rank_0("> finished creating MAE datasets ...")

    return train_ds, None, None


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )
