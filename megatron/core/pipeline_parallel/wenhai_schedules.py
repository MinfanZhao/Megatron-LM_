# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager, nullcontext
from typing import Optional, List, Union, Callable, Any

import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication, wenhai_p2p_communication
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model, get_model_type

# Types
Shape = Union[List[int], torch.Size]

def get_forward_backward_func(is_finetune=False):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func

    model (required): the actual model. A torch.nn.Module or, in the
        case or iterleaving, a list of torch.nn.Module

    num_microbatches (int, required):
        The number of microbatches to go through

    dtype (required when using pipeline parallelism): dtype used in
        p2p communication, usually params_dtype

    tensor_shape (required when using pipeline parallelism): Shape of
        tensor. The tensor is expected to be 3D and its order of
        dimension is supposed to be ``(sequence, batch, hidden)``.

    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    grad_scaler (optional, default=None): If using loss scaling,
        this function should take the loss and return the scaled
        loss. If None, no function is called on the loss.

    sequence_parallel (optional, default=False):
        Set to :obj:`True` for this function to handle sequence
        length.  When :obj:`True`, the sequence length on each tensor
        model parallel rank is updated to
        :math:`original\_sequence\_length /
        tensor\_model\_parallel\_world\_size`.
        TODO: Do we need this? Just roll into tensor_shape arg?

    forward_only (optional, default=False): Perform only the forward step

    timers (optional, default=None): TODO

    collect_non_loss_data: TODO

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None
        if is_finetune:
            forward_backward_func = forward_backward_pipelining_wenhai
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
        
    return forward_backward_func

def deallocate_output_tensor(out):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out is None:
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )

def custom_backward(output, grad_output, keep_graph=False):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = keep_graph,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(forward_step_func,
                 data,
                 model,
                 num_microbatches,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False,
                 day=0,
                 is_finetune=False,
                 loss_weight=None):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if timers is not None:
        timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    context_manager = torch.autocast("cuda") if torch.is_autocast_enabled() else nullcontext()
    with context_manager:
        if is_finetune:
            output_tensor, loss_func = forward_step_func(data, model, day=day)
        else:
            output_tensor, loss_func = forward_step_func(data, model)
    if is_finetune:
        output_tensor = output_tensor.half()
    # print(f"in forward step {output_tensor.shape} {output_tensor.dtype}")

    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor, loss_weight)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, loss_weight, non_loss_data=True)
            forward_data_store.append(data)

    if timers is not None:
        timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    # print(f"[schedules forward_step] output_tensor shape {output_tensor.shape}")

    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(grad_scaler, input_tensor, output_tensor,
                  output_tensor_grad, model_type, timers, keep_graph=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor = grad_scaler(output_tensor[0])
    custom_backward(output_tensor[0], output_tensor_grad[0], keep_graph=keep_graph)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


def get_tensor_shapes(*,
                      rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    seq_length, micro_batch_size, hidden_size = tensor_shape

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if model_type == ModelType.encoder_and_decoder:
        if sequence_parallel:
            decoder_seq_length = decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()

        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    return tensor_shapes


def get_wenhai_tensor_shapes(*,
                      rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      result_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool,
                      use_result_shape: bool=False):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []
    assert model_type ==  ModelType.encoder_or_decoder
    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    micro_batch_size, seq_length, hidden_size = tensor_shape

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if use_result_shape:
        tensor_shapes.append(result_shape)
    else:
        tensor_shapes.append((micro_batch_size, seq_length, hidden_size))
    return tensor_shapes


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def recv_forward(tensor_shapes, dtype, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, dtype,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, dtype, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, dtype,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, timers=timers)


def send_forward_recv_backward(output_tensors, tensor_shapes, dtype, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, dtype, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, dtype, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, dtype, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors



def wenhai_recv_forward(tensor_shapes, dtype, timers, is_first_day):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(wenhai_p2p_communication.wenhai_recv_forward(tensor_shape, dtype,
                                                                timers=timers, is_first_day=is_first_day))
            
    return input_tensors


def wenhai_recv_backward(tensor_shapes, dtype, timers, is_last_day):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(wenhai_p2p_communication.wenhai_recv_backward(tensor_shape, dtype,
                                                                       timers=timers, is_last_day=is_last_day))
    return output_tensor_grads


def wenhai_send_forward(output_tensors, tensor_shapes, timers, is_last_day):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        wenhai_p2p_communication.wenhai_send_forward(output_tensor, timers=timers, is_last_day=is_last_day)


def wenhai_send_backward(input_tensor_grads, tensor_shapes, timers, is_first_day):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        wenhai_p2p_communication.wenhai_send_backward(input_tensor_grad, timers=timers, is_first_day=is_first_day)


def forward_backward_pipelining_wenhai(*,
                                                     forward_step_func,
                                                     data,
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     dtype: torch.dtype,
                                                     tensor_shape: Shape,
                                                     result_shape: Shape,
                                                     decoder_seq_length: Optional[int] = None,
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False,
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = True, 
                                                     total_day: int = 8,
                                                     loss_weight: torch.Tensor = None):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    
    assert num_microbatches == 1, "num micro batch > 1 is not implemented in wenhai pipeline"
    assert len(model) == 1
    assert total_day > 1
    model = model[0]
    # print("<<<<<<<<<<<<<<<<< running wenhai forward backward pipeline")

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)               # num_warmup_microbatches = 1
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches          # num_microbatches_remaining = 0

    # print(f"num_warmup_microbatches:{num_warmup_microbatches} num_microbatches_remaining:{num_microbatches_remaining}")

    model_type = get_model_type(model)

    
    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_wenhai_tensor_shapes(rank=rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           result_shape=result_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel,
                                           use_result_shape=parallel_state.is_pipeline_first_stage())
    send_tensor_shapes = get_wenhai_tensor_shapes(rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           result_shape=result_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel,
                                           use_result_shape=parallel_state.is_pipeline_last_stage())
    
    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    
    send_loss_tensor_shapes = [(1)]
    recv_loss_tensor_shapes = [(1)]
    # loss_tensor = torch.Tensor([0]).half().cuda()
    
    
    ###################### First day ########################
    input_tensor = wenhai_recv_forward(recv_tensor_shapes, dtype, timers=timers, is_first_day=True)
    [loss_tensor] = wenhai_recv_forward(recv_loss_tensor_shapes, dtype, timers=timers, is_first_day=True)
    output_tensor = forward_step(forward_step_func, data, model, num_microbatches, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data, day=0, is_finetune=True, loss_weight=loss_weight)
    
    if parallel_state.is_pipeline_last_stage():
        (new_loss_tensor, new_loss_reduced) = forward_data_store[-1]
        new_loss_tensor = loss_tensor + new_loss_tensor
        loss_tensor = new_loss_tensor.view([1])
        print(f"day 1 / {total_day} forward done! loss is {loss_tensor}")

    
    wenhai_send_forward(output_tensor, send_tensor_shapes, timers=timers, is_last_day=False)

    if loss_tensor is None:
        loss_tensor = torch.Tensor([0]).half().cuda()
    wenhai_send_forward([loss_tensor], send_loss_tensor_shapes, timers=timers, is_last_day = False)


    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor[0])

    ##################### Remaining days #########################
    for day in range(1, total_day):
        input_tensor = wenhai_recv_forward(recv_tensor_shapes, dtype, timers=timers, is_first_day = False)
        [loss_tensor] = wenhai_recv_forward(recv_loss_tensor_shapes, dtype, timers=timers, is_first_day=False)
        output_tensor = forward_step(forward_step_func, data, model, num_microbatches, input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, day=day, is_finetune=True, loss_weight=loss_weight)
        if parallel_state.is_pipeline_last_stage():
            (new_loss_tensor, new_loss_reduced) = forward_data_store[-1]
            new_loss_tensor = loss_tensor + new_loss_tensor
            loss_tensor = new_loss_tensor.view([1])
            print(f"day {day + 1} / {total_day} forward done! loss is {loss_tensor}")
        
        wenhai_send_forward(output_tensor, send_tensor_shapes, timers=timers, is_last_day = total_day == day + 1)
        wenhai_send_forward([loss_tensor], send_loss_tensor_shapes, timers=timers, is_last_day = total_day == day + 1)
        

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])
            
            
        if total_day == day + 1 and parallel_state.is_pipeline_last_stage() and not forward_only:
            output_tensors[-1] = [loss_tensor.view([])]

    # Run cooldown backward passes.
    if not forward_only:
        # for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop()
        output_tensor = output_tensors.pop()

        output_tensor_grad = wenhai_recv_backward(send_tensor_shapes, dtype, timers=timers, is_last_day=True)

        input_tensor_grad = \
            backward_step(grad_scaler, input_tensor, output_tensor,
                            output_tensor_grad, model_type, timers, keep_graph=True)

        wenhai_send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers, is_first_day = total_day == 1)
    
        for day in range(total_day - 2, -1, -1):
            # for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop()
            output_tensor = output_tensors.pop()

            output_tensor_grad = wenhai_recv_backward(send_tensor_shapes, dtype, timers=timers, is_last_day=False)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                            output_tensor_grad, model_type, timers, keep_graph=True)

            wenhai_send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers, is_first_day = day == 0)
            
    clean_finetune_buffer = get_attr_wrapped_model(model, "clean_finetune_buffer")
    clean_finetune_buffer()

    return forward_data_store


def forward_backward_no_pipelining(*,
                                   forward_step_func,
                                   data_iterator,
                                   model: Union[torch.nn.Module, List[torch.nn.Module]],
                                   num_microbatches: int,
                                   dtype: Optional[torch.dtype] = None, # unused
                                   tensor_shape: Optional[Shape] = None, # unused
                                   decoder_seq_length: Optional[int] = None, # unused
                                   grad_scaler: Callable = None,
                                   sequence_parallel: bool = False, # unused
                                   forward_only: bool = False,
                                   timers: Callable = None,
                                   collect_non_loss_data: bool = False):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, num_microbatches, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data)
            if not forward_only:
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, num_microbatches, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data)

    if not forward_only:
        backward_step(grad_scaler, input_tensor, output_tensor,
                      output_tensor_grad, model_type, timers)
    
    clean_finetune_buffer = get_attr_wrapped_model(model, "clean_finetune_buffer")
    clean_finetune_buffer()
    return forward_data_store


def forward_backward_pipelining_without_interleaving(*,
                                                     forward_step_func,
                                                     data_iterator,
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     dtype: torch.dtype,
                                                     tensor_shape: Shape,
                                                     decoder_seq_length: Optional[int] = None,
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False,
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = False):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank=rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    # print(f"[schedules forward_backward_pipelining_without_interleaving] tensor shape {recv_tensor_shapes} {send_tensor_shapes}")
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
        # print(f"rank:{parallel_state.get_pipeline_model_parallel_rank()} input_tensor:{input_tensor}")
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,input_tensor, forward_data_store,timers, collect_non_loss_data)
        send_forward(output_tensor, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])
            # print(f"rank:{parallel_state.get_pipeline_model_parallel_rank()} output_tensors:{output_tensors}")

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data)

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, timers=timers)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)

        else:
            output_tensor_grad = \
                send_forward_recv_backward(output_tensor,
                                           send_tensor_shapes, dtype,
                                           timers=timers)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers)

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            else:
                input_tensor = \
                    send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, dtype, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, dtype, timers=timers)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers)

            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
    clean_finetune_buffer = get_attr_wrapped_model(model, "clean_finetune_buffer")
    clean_finetune_buffer()

    return forward_data_store