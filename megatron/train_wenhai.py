# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

from functools import partial
import sys
import torch
import time
import math
from datetime import datetime
_TRAIN_START_TIME = time.time()

from megatron import get_args, get_num_microbatches
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import setup_model_and_optimizer
from megatron.utils import calc_params_l2_norm
from megatron.utils import check_adlr_autoresume_termination
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron import get_tensorboard_writer

from megatron import is_last_rank

from megatron import print_rank_last
from megatron.utils import report_memory

from megatron.core.pipeline_parallel import wenhai_get_forward_backward_func

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))



def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None, shuffle=True):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=micro_batch_size,
                                            sampler=sampler,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            drop_last=drop_last,
                                            pin_memory=False,
                                            collate_fn=task_collate_fn)

    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, 
    task_collate_fn=None):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size
    
    train_dataloader, valid_dataloader = None, None
    if mpu.get_tensor_model_parallel_rank() == 0:
        train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
                                            args.num_workers, True,
                                            task_collate_fn)
        # Validation dataset. For this dataset, we do not need to set up
        # shuffling so we can just use a simple infinite loop.
        valid_dataloader = build_data_loader(valid_dataset, args.micro_batch_size,
                                            args.num_workers, False,
                                            task_collate_fn)
        # valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

        # Set the training iterations.
        args.train_iters_per_epoch = len(train_dataloader) // get_num_microbatches()
        # Now that we've built the data loaders, set batch_size arguments
        # to the actual batch size the model will see for this dataset.
        # This is necessary so pipeline transfers know what size they are
        # and the LR schedule, which is based on samples seen, gets set
        # correctly.
        
        if hasattr(train_dataset, 'sample_multiplier'):
            # If our dataset as a sample_multiplier attribute that means
            # each "sample" from the dataset actually has multiple samples
            # that will collapse into the batch dimension (for example in
            # the RACE dataset that has several options), we need to
            # account for that when setting the micro batch size.
            args.micro_batch_size *= train_dataset.sample_multiplier
            args.global_batch_size *= train_dataset.sample_multiplier
        share_args = torch.cuda.LongTensor(
            [int(args.train_iters_per_epoch), int(args.micro_batch_size), int(args.global_batch_size)])
    else:
        share_args = torch.cuda.LongTensor([0, 0, 0])
    torch.distributed.broadcast(share_args,
                            mpu.get_tensor_model_parallel_src_rank(),
                            group=mpu.get_tensor_model_parallel_group())
    args.train_iters_per_epoch = share_args[0].item()
    args.micro_batch_size = share_args[1].item()
    args.global_batch_size = share_args[2].item()
    
    args.train_iters = args.epochs * args.train_iters_per_epoch
    return train_dataloader, valid_dataloader


def _train(model, optimizer, opt_param_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback, finetune=False, get_batch_func=None):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # assert get_num_microbatches() == 1, "finetuning with gradient accumulation doesn't currently work"

    # Turn on training mode which enables dropout.
    for m in model:
        m.train()

    # Tracking loss.
    losses_dict_sum = {}
    if finetune:
        loss_weight = torch.load('/test2/cuiyz/data/weight_Sx4.5.pt').half().cuda()
    else:
        loss_weight = None
    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers('interval-time', log_level=0).start(barrier=True)
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        if mpu.get_tensor_model_parallel_rank() == 0:
            train_dataloader.sampler.set_epoch(args.seed + epoch)
            data_iterator = iter(train_dataloader)
            valid_dataloader.sampler.set_epoch(args.seed + epoch)
            valid_data_iterator = iter(valid_dataloader)
        else:
            data_iterator = None
        # For all the batches in the dataset.
        for iteration_ in range(args.train_iters_per_epoch):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            out = train_step(forward_step, get_batch_func, data_iterator, model, optimizer, 
                             opt_param_scheduler, is_finetune=finetune, loss_weight=loss_weight)

            losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = out
            iteration += 1
            args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()

            # Logging.
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration,
                                              optimizer.get_loss_scale().item(),
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad)

            # Autoresume
            if args.adlr_autoresume and \
               (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model,
                                                  optimizer, opt_param_scheduler)

            # Checkpointing
            saved_checkpoint = False
            if args.save and args.save_interval and \
               iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
                saved_checkpoint = True

            # ##################### DEBUG ###############
            # sys.exit()

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                prefix = 'iteration {}'.format(iteration)
                if finetune:
                    wenhai_evaluate_and_print_results(prefix, forward_step, get_batch_func,
                                            valid_data_iterator, model,
                                            iteration, None, True, is_finetune=True, loss_weight=loss_weight)
                else:
                    wenhai_evaluate_and_print_results(prefix, forward_step,
                                            valid_data_iterator, model,
                                            iteration, None, True, is_finetune=False)

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if not saved_checkpoint:
                    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
                torch.distributed.barrier()
                print_rank_0('exiting program at iteration {}'.format(iteration))
                sys.exit()

        # Checkpointing at the end of each epoch.
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch)


def pretrain_by_epoch(train_valid_datasets_provider, model_provider,
             model_type=ModelType.encoder_or_decoder,
             forward_step=None,
             end_of_epoch_callback_provider=None,
             task_collate_fn=None,
             extra_args_provider=None,
             args_defaults={}):
    
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()
    
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    
    args = get_args()
    timers = get_timers()

    assert args.rampup_batch_size is None, \
        'batch size scaling is not supported for pretrain_by_epoch'

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder', log_level=0).start()
    if args.epochs > 0:
        if mpu.get_tensor_model_parallel_rank() == 0:
            train_dataset, valid_dataset = train_valid_datasets_provider()
        else:
            train_dataset, valid_dataset = None, None
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn)
    else:
        args.train_iters = 0
    timers('train/valid/test dataset/dataloder').stop()

    # Build calback function.
    timers('callback function', log_level=0).start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer', log_level=0).start()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint', log_level=0).start(barrier=True)
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        original_rng = args.no_load_rng
        args.no_load_rng = True
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        args.no_load_rng = original_rng
        # This is critical when only model is loaded. We should make sure
        # main parameters are also updated.
        optimizer.reload_model_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'], barrier=True)
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model, optimizer, opt_param_scheduler, forward_step,
               train_dataloader, valid_dataloader, end_of_epoch_callback)
    # Or just evaluate.
    else:
        if end_of_epoch_callback is not None:
            print_rank_0('evaluation only mode, setting epoch to -1')
            end_of_epoch_callback(model, epoch=-1, output_predictions=True)
    print_rank_0('done :-)')



def finetune_by_epoch(train_valid_datasets_provider, model_provider,
             model_type=ModelType.encoder_or_decoder,
             forward_step=None,
             get_batch=None,
             end_of_epoch_callback_provider=None,
             task_collate_fn=None,
             extra_args_provider=None,
             args_defaults={}):
    
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()
    
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    
    args = get_args()
    timers = get_timers()
 
    assert args.rampup_batch_size is None, \
        'batch size scaling is not supported for finetune_by_epoch'

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder', log_level=0).start()
    if args.epochs > 0:
        if mpu.get_tensor_model_parallel_rank() == 0:
            train_dataset, valid_dataset = train_valid_datasets_provider()
        else:
            train_dataset, valid_dataset = None, None
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn)
    else:
        args.train_iters = 0
    timers('train/valid/test dataset/dataloder').stop()

    # Build calback function.
    timers('callback function', log_level=0).start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer', log_level=0).start()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint', log_level=0).start(barrier=True)
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        original_rng = args.no_load_rng
        args.no_load_rng = True
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        args.no_load_rng = original_rng
        # This is critical when only model is loaded. We should make sure
        # main parameters are also updated.
        optimizer.reload_model_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'], barrier=True)
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model, optimizer, opt_param_scheduler, forward_step,
               train_dataloader, valid_dataloader, end_of_epoch_callback, finetune=True, get_batch_func=get_batch)
    # Or just evaluate.
    else:
        if end_of_epoch_callback is not None:
            print_rank_0('evaluation only mode, setting epoch to -1')
            end_of_epoch_callback(model, epoch=-1, output_predictions=True)
    print_rank_0('done :-)')
    
    
def train_step(forward_step_func, get_batch_func, data_iterator,
               model, optimizer, opt_param_scheduler, is_finetune=False, loss_weight=None):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    if is_finetune:
        timers('batch-generator', log_level=2).start()
        data = get_batch_func(data_iterator)
        timers('batch-generator').stop()
        forward_backward_func = wenhai_get_forward_backward_func(is_finetune=True)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data=data,
            model=model,
            num_microbatches=get_num_microbatches(),
            dtype=args.params_dtype,
            tensor_shape=(args.micro_batch_size, args.seq_length, args.hidden_size),
            result_shape=(args.micro_batch_size, args.out_channels, args.img_size[0], args.img_size[1]),
            grad_scaler=optimizer.scale_loss,
            sequence_parallel=args.sequence_parallel,
            forward_only=False,
            timers=fwd_bwd_timers,
            total_day=args.finetune_days,
            loss_weight=loss_weight)
    else:
        forward_backward_func = wenhai_get_forward_backward_func(is_finetune=False)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            dtype=args.params_dtype,
            tensor_shape=(args.micro_batch_size, args.seq_length, args.hidden_size) if args.bsh_tensor_shape \
                else (args.seq_length, args.micro_batch_size, args.hidden_size),
            grad_scaler=optimizer.scale_loss,
            sequence_parallel=args.sequence_parallel,
            forward_only=False,
            timers=fwd_bwd_timers)
    timers('forward-backward').stop()

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    optimizer.reduce_model_grads(args, timers)


    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful:
        optimizer.gather_model_params(args, timers)


    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        if is_finetune:
            losses = []
            for loss_value in losses_reduced:
                loss_value = loss_value[1]
                if isinstance(loss_value, dict):
                    losses.append(loss_value["loss"])
            loss_reduced["loss"] = sum(losses)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        else:
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad
    

def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    
    
    # print(f"loss dict:{loss_dict}")
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            # print(f"key:{key}, loss_dict[key]:{loss_dict[key]}")
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag



def evaluate(forward_step_func,
             get_batch_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             verbose=False, is_finetune=False, loss_weight=None):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()
    loss_reduced = {}
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            timers('forward-backward', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            fwd_bwd_timers = timers if args.timing_log_level > 1 else None
            if is_finetune:
                timers('batch-generator', log_level=2).start()
                data = get_batch_func(data_iterator)
                timers('batch-generator').stop()
                forward_backward_func = wenhai_get_forward_backward_func(is_finetune=True)
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data=data,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    dtype=args.params_dtype,
                    tensor_shape=(args.micro_batch_size, args.seq_length, args.hidden_size),
                    result_shape=(args.micro_batch_size, args.out_channels, args.img_size[0], args.img_size[1]),
                    sequence_parallel=args.sequence_parallel,
                    forward_only=True,
                    timers=fwd_bwd_timers,
                    total_day=args.finetune_days,
                    loss_weight=loss_weight)
                print_rank_last(f"eval iter:{iteration} | forward backward done")
            else:
                forward_backward_func = wenhai_get_forward_backward_func(is_finetune=False)
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    dtype=args.params_dtype,
                    tensor_shape=(args.micro_batch_size, args.seq_length, args.hidden_size) if args.bsh_tensor_shape \
                        else (args.seq_length, args.micro_batch_size, args.hidden_size),
                    sequence_parallel=args.sequence_parallel,
                    forward_only=True,
                    timers=fwd_bwd_timers)
            timers('forward-backward').stop()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                
                if is_finetune:
                    losses = []
                    for loss_value in losses_reduced:
                        loss_value = loss_value[1]
                        if isinstance(loss_value, dict):
                            losses.append(loss_value["loss"])
                    # print(f"losses:{losses}")
                    if loss_reduced.__contains__("loss"):
                        loss_reduced["loss"] += sum(losses)
                    else:
                        loss_reduced["loss"] = sum(losses)

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        # if process_non_loss_data_func is not None and is_last_rank():
        #     collected_non_loss_data = forward_backward_func(
        #         forward_step_func, data_iterator, model, optimizer=None,
        #         timers=None, forward_only=True, collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in loss_reduced:
        if is_finetune:
            print(f"loss reduced: {loss_reduced}")
            loss_reduced[key] /= args.eval_iters * get_num_microbatches()
        else:
            loss_reduced[key] /= args.eval_iters * get_num_microbatches()

    return loss_reduced, collected_non_loss_data



def wenhai_evaluate_and_print_results(prefix, forward_step_func, get_batch_func, 
                               data_iterator, model,
                               iteration, process_non_loss_data_func,
                               verbose=False, is_finetune = False, loss_weight = None):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, get_batch_func, data_iterator, model,
        process_non_loss_data_func, verbose, is_finetune, loss_weight)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)