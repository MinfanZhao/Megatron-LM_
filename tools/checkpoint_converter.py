# checkpoint converter
import argparse
import os
import torch
import time
import sys

def print_with_time(s):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {s}')


def check_keys(model_dict, print_shape=True):
    print_with_time(model_dict.keys())
    for key in model_dict.keys():
        if isinstance(model_dict[key], torch.Tensor):
            if print_shape:
                print_with_time(f"{key}:{model_dict[key].shape}")
            else:
                print_with_time(f"{key}")
        if isinstance(model_dict[key], dict):
            check_keys(model_dict[key])



def get_weight_name(weight_name, layer_index=None):
    if layer_index is None:
        weight_name_dict = {
            'final_norm.weight': 'final_layernorm.weight',
            'final_norm.bias': 'final_layernorm.bias',
            'output_layer': 'output_layer',
        }
    else:
        weight_name_dict = {
            'norm_1.weight' : f'layers.{layer_index}.input_layernorm.weight',
            'norm_1.bias' : f'layers.{layer_index}.input_layernorm.bias',
            'norm_2.weight': f'layers.{layer_index}.post_attention_layernorm.weight',
            'norm_2.bias': f'layers.{layer_index}.post_attention_layernorm.bias',
            'qkv.weight' : f'layers.{layer_index}.self_attention.query_key_value.weight',
            'qkv.bias' : f'layers.{layer_index}.self_attention.query_key_value.bias',
            'dense.weight': f'layers.{layer_index}.self_attention.dense.weight',
            'dense.bias': f'layers.{layer_index}.self_attention.dense.bias',
            'mlp_up.weight': f'layers.{layer_index}.mlp.dense_h_to_4h.weight',
            'mlp_up.bias': f'layers.{layer_index}.mlp.dense_h_to_4h.bias',
            'mlp_down.weight': f'layers.{layer_index}.mlp.dense_4h_to_h.weight', 
            'mlp_down.bias': f'layers.{layer_index}.mlp.dense_4h_to_h.bias', 
        }

    return weight_name_dict[weight_name]


def pad_vocab(word_embeddings, output_layer, hidden_size,
              orig_vocab_size, vocab_size_divisible_by, tensor_model_parallel_size):
    
    def _vocab_size_with_padding(orig_vocab_size, vocab_size_divisible_by, tensor_model_parallel_size):
        after = orig_vocab_size
        multiple = vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        print_with_time(' > padded vocab (size: {}) with {} dummy tokens '
                '(new size: {})'.format(
                    orig_vocab_size, after - orig_vocab_size, after))
        return after
    
    if output_layer is not None:
        assert word_embeddings.shape[0] == output_layer.shape[0] 
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, vocab_size_divisible_by, tensor_model_parallel_size)
    current_vocab_size = word_embeddings.shape[0]
    
    if padded_vocab_size > current_vocab_size:
        word_embeddings = torch.cat([word_embeddings, torch.zeros(padded_vocab_size-current_vocab_size, hidden_size, dtype=word_embeddings.dtype)], dim=0)
        if output_layer is not None:
            output_layer = torch.cat([output_layer, torch.zeros(padded_vocab_size-current_vocab_size, hidden_size, dtype=output_layer.dtype)], dim=0)
    else:
        word_embeddings = word_embeddings[:padded_vocab_size,:].clone()
        if output_layer is not None:
            output_layer = output_layer[:padded_vocab_size,:].clone()
    return word_embeddings, output_layer
    

def get_checkpoint_base(input_dir):
    tracker_filename = os.path.join(input_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'r') as f:
        iteration = f.read().strip()
        if iteration != 'release':
            iteration = int(iteration)
            megatron_checkpoint_base = os.path.join(
                input_dir, 'iter_{:07d}'.format(iteration))
        else:
            megatron_checkpoint_base = os.path.join(
                input_dir, 'release')
    return megatron_checkpoint_base


def load_checkpoint(input_dir, num_layers, hidden_size, add_bias_linear, 
                    untie_embeddings_and_output_weights, gate_gelu,
                    load_pipeline_model_parallel_size, 
                    load_tensor_model_parallel_size):
    
    
    # def check_args(args):
        
    #     assert args.num_layers == num_layers or num_layers is None
    #     assert args.hidden_size == hidden_size or hidden_size
    #     assert args.add_bias_linear == add_bias_linear
    #     assert args.untie_embeddings_and_output_weights == untie_embeddings_and_output_weights
    #     assert args.pipeline_model_parallel_size == load_pipeline_model_parallel_size
    #     assert args.tensor_model_parallel_size == load_tensor_model_parallel_size
    
    checkpoint_base = get_checkpoint_base(input_dir)
    print_with_time(f'checkpoint base: {checkpoint_base}')

    # load checkpoint from megatron
    state_dict_shards = []
    pipeline_parallel = load_pipeline_model_parallel_size > 1
    for pipeline_rank in range(load_pipeline_model_parallel_size):
        state_dict_shards.append([])
        is_first_stage = True if pipeline_rank == 0 else False
        is_last_stage = True if pipeline_rank == load_pipeline_model_parallel_size - 1 else False
            
        for tensor_rank in range(load_tensor_model_parallel_size):
            if not pipeline_parallel:
                common_path = os.path.join(checkpoint_base, 
                                f'mp_rank_{tensor_rank:02d}')
            else:
                common_path = os.path.join(checkpoint_base, 
                                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
            model_checkpoint_name = os.path.join(common_path, 'model_optim_rng.pt')
            print_with_time(f'loading {model_checkpoint_name} ...')
            state_dict_shard = torch.load(model_checkpoint_name, map_location='cpu')

            language_model = state_dict_shard['model']['language_model']
            # if is_last_stage and not untie_embeddings_and_output_weights:
            #     # move word_embeddings_for_head to language_model.output_layer for simplicity
            #     if not is_first_stage:
            #         language_model['output_layer'] = {'weight': state_dict_shard['model']['word_embeddings_for_head']['weight']}

            state_dict_shards[pipeline_rank].append(language_model)
            print_with_time(f'load {model_checkpoint_name} done.')
    
    layer_per_partition = num_layers // load_pipeline_model_parallel_size
    assert layer_per_partition * load_pipeline_model_parallel_size == num_layers     
            
    # merge state_dict_shards
    embedding = {'word_embeddings':{'weight':None}}
    encoder = {}
    output_layer = {'weight':None}
    
    for pipeline_rank in range(load_pipeline_model_parallel_size):
        is_first_stage = True if pipeline_rank == 0 else False
        is_last_stage = True if pipeline_rank == load_pipeline_model_parallel_size - 1 else False
        layer_offset = layer_per_partition * pipeline_rank

        shard_encoder = state_dict_shards[pipeline_rank][0]['encoder']
        
        # check_keys(state_dict_shards[pipeline_rank][0], print_shape=False)
        
        # merge encoder
        for partition_index in range(layer_per_partition):
            layer_index = layer_offset + partition_index
            
            # layernorm
            encoder[get_weight_name('norm_1.weight', layer_index)] = shard_encoder[get_weight_name('norm_1.weight', partition_index)] 
            encoder[get_weight_name('norm_2.weight', layer_index)] = shard_encoder[get_weight_name('norm_2.weight', partition_index)] 
            
            if add_bias_linear:
                encoder[get_weight_name('norm_1.bias', layer_index)] = shard_encoder[get_weight_name('norm_1.bias', partition_index)] 
                encoder[get_weight_name('norm_2.bias', layer_index)] = shard_encoder[get_weight_name('norm_2.bias', partition_index)] 
            

            # attention
            encoder[get_weight_name('qkv.weight', layer_index)] = \
                  torch.cat([tensor_shard['encoder'][get_weight_name('qkv.weight', partition_index)] 
                             for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
            # \
                # torch.cat([tensor_shard['encoder'][get_weight_name('qkv.weight', partition_index)].view(
                #     1, -1, hidden_size) for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(
                #     -1, hidden_size)

            encoder[get_weight_name('dense.weight', layer_index)] = \
                torch.cat([tensor_shard['encoder'][get_weight_name('dense.weight', partition_index)].view(
                    hidden_size, 1, -1) for tensor_shard in state_dict_shards[pipeline_rank]], dim=1).view(
                    hidden_size, hidden_size)
                    
            if  add_bias_linear:
                encoder[get_weight_name('qkv.bias', layer_index)] = \
                    torch.cat([tensor_shard['encoder'][get_weight_name('qkv.bias', partition_index)].view(
                        1, -1) for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(-1)
                encoder[get_weight_name('dense.bias', layer_index)] = shard_encoder[get_weight_name('dense.bias', partition_index)]

            # ffn
            if gate_gelu:
                encoder[get_weight_name('mlp_up.weight', layer_index)] = \
                    torch.cat([tensor_shard['encoder'][get_weight_name('mlp_up.weight', partition_index)].view(
                        -1, 2, hidden_size) for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(-1, hidden_size)
            else:
                encoder[get_weight_name('mlp_up.weight', layer_index)] = \
                    torch.cat([tensor_shard['encoder'][get_weight_name('mlp_up.weight', partition_index)] 
                                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
            
            encoder[get_weight_name('mlp_down.weight', layer_index)] = \
                torch.cat([tensor_shard['encoder'][get_weight_name('mlp_down.weight', partition_index)] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=1)
            
            if add_bias_linear:
                if gate_gelu:
                    encoder[get_weight_name('mlp_up.bias', layer_index)] = \
                        torch.cat([tensor_shard['encoder'][get_weight_name('mlp_up.bias', partition_index)].view(
                            -1, 2) for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(-1)
                    # encoder[get_weight_name('mlp_up.bias', layer_index)] = \
                    #     torch.cat([tensor_shard['encoder'][get_weight_name('mlp_up.bias', partition_index)].view(
                    #         2, -1) for tensor_shard in state_dict_shards[pipeline_rank]], dim=1).view(-1)
                else:
                    encoder[get_weight_name('mlp_up.bias', layer_index)] = \
                        torch.cat([tensor_shard['encoder'][get_weight_name('mlp_up.bias', partition_index)] \
                                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
                encoder[get_weight_name('mlp_down.bias', layer_index)] = shard_encoder[get_weight_name('mlp_down.bias', partition_index)]

        # embedding
        if is_first_stage:
            try:
                embedding['word_embeddings']['weight'] = \
                torch.cat([tensor_shard['embedding']['word_embeddings']['weight'] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
            except:
                embedding['word_embeddings']['weight'] = \
                torch.cat([tensor_shard['embedding']['word_embeddings.weight'] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)

        # final norm & output layer
        if is_last_stage:
            encoder[get_weight_name('final_norm.weight')] = shard_encoder[get_weight_name('final_norm.weight')]
            encoder[get_weight_name('final_norm.bias')] = shard_encoder[get_weight_name('final_norm.bias')]
            if untie_embeddings_and_output_weights:
                output_layer['weight'] = \
                    torch.cat([tensor_shard[get_weight_name('output_layer')]['weight'] \
                               for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)

    if untie_embeddings_and_output_weights:
        return {
            'language_model':{
                'embedding': embedding,
                'encoder': encoder,
                'output_layer': output_layer,
            } 
        }
    else:
        return {
            'language_model':{
                'embedding': embedding,
                'encoder': encoder,
            }
        }

            
    

def save_checkpoint(whole_state_dict, output_dir, num_layers, hidden_size, 
                    num_attention_heads, add_bias_linear, 
                    untie_embeddings_and_output_weights, gate_gelu,
                    vocab_size, make_vocab_size_divisible_by,
                    save_pipeline_model_parallel_size, 
                    save_tensor_model_parallel_size):
    
   
    
    def split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank):
        is_first_stage = True if pipeline_rank == 0 else False
        is_last_stage = True if pipeline_rank == save_pipeline_model_parallel_size - 1 else False
        splited_state_dict = {"language_model":{"encoder": {}}}
        layer_offset = layer_per_partition * pipeline_rank

        encoder = whole_state_dict['language_model']['encoder']
        shard_encoder = {}
        
        for partition_index in range(layer_per_partition):
            layer_index = layer_offset + partition_index


            # layernorm
            shard_encoder[get_weight_name('norm_1.weight', partition_index)] = encoder[get_weight_name('norm_1.weight', layer_index)].clone()
            shard_encoder[get_weight_name('norm_2.weight', partition_index)] = encoder[get_weight_name('norm_2.weight', layer_index)].clone()

            if add_bias_linear:
                shard_encoder[get_weight_name('norm_1.bias', partition_index)] = encoder[get_weight_name('norm_1.bias', layer_index)].clone()
                shard_encoder[get_weight_name('norm_2.bias', partition_index)] = encoder[get_weight_name('norm_2.bias', layer_index)].clone()

            # attention
            qkv = encoder[get_weight_name('qkv.weight', layer_index)]
            qkv_shape = qkv.shape
            shard_qkv = qkv.view(save_tensor_model_parallel_size, -1, qkv_shape[1])[tensor_rank,:,:]
            shard_encoder[get_weight_name('qkv.weight', partition_index)] = shard_qkv.reshape(-1, qkv_shape[1]).clone()
                
            dense = encoder[get_weight_name('dense.weight', layer_index)]
            shard_encoder[get_weight_name('dense.weight', partition_index)] = \
                dense.view(dense.shape[0],save_tensor_model_parallel_size, -1)[:,tensor_rank,:].clone()
            
            if add_bias_linear:
                qkv_bias = encoder[get_weight_name('qkv.bias', layer_index)]
                shard_qkv_bias = qkv_bias.view(save_tensor_model_parallel_size, -1)[tensor_rank,:]
                shard_encoder[get_weight_name('qkv.bias', partition_index)] = shard_qkv_bias.reshape(-1).clone()
                shard_encoder[get_weight_name('dense.bias', partition_index)] = encoder[get_weight_name('dense.bias', layer_index)].clone()
                

            # ffn   
            mlp_up = encoder[get_weight_name('mlp_up.weight', layer_index)]
            if gate_gelu:
                # shard_mlp_up = mlp_up.view(2, save_tensor_model_parallel_size, -1, mlp_up.shape[1])[:,tensor_rank,:,:]
                shard_mlp_up = mlp_up.view(save_tensor_model_parallel_size, 2, -1, mlp_up.shape[1])[tensor_rank,:,:,:]
                # shard_mlp_up = mlp_up.view(save_tensor_model_parallel_size, -1, 2, mlp_up.shape[1])[tensor_rank,:,:,:]

            else:
                shard_mlp_up = mlp_up.view(save_tensor_model_parallel_size, -1, mlp_up.shape[1])[tensor_rank,:,:]
            shard_encoder[get_weight_name('mlp_up.weight', partition_index)] = shard_mlp_up.reshape(-1, mlp_up.shape[1]).clone()
                
                
            mlp_down = encoder[get_weight_name('mlp_down.weight', layer_index)]
            shard_encoder[get_weight_name('mlp_down.weight', partition_index)] = \
                mlp_down.view(mlp_down.shape[0], save_tensor_model_parallel_size, -1)[:,tensor_rank,:].clone()

            if add_bias_linear:
                mlp_up_bias = encoder[get_weight_name('mlp_up.bias', layer_index)]
                if gate_gelu:
                    # shard_mlp_up_bias = mlp_up_bias.view(2, save_tensor_model_parallel_size, -1)[:,tensor_rank,:]
                    shard_mlp_up_bias = mlp_up_bias.view(save_tensor_model_parallel_size, 2, -1)[tensor_rank,:,:]
                    # shard_mlp_up_bias = mlp_up_bias.view(save_tensor_model_parallel_size, -1, 2)[tensor_rank,:,:]
                else:
                    shard_mlp_up_bias = mlp_up_bias.view(save_tensor_model_parallel_size, -1)[tensor_rank,:]
                shard_encoder[get_weight_name('mlp_up.bias', partition_index)] = shard_mlp_up_bias.reshape(-1).clone()
                shard_encoder[get_weight_name('mlp_down.bias', partition_index)] = encoder[get_weight_name('mlp_down.bias', layer_index)].clone()

          
        
        if is_first_stage:
            word_embeddings = whole_state_dict['language_model']['embedding']['word_embeddings']['weight']
            splited_word_embedding = word_embeddings.view(save_tensor_model_parallel_size, -1 ,word_embeddings.shape[1])[tensor_rank,:,:]
            splited_state_dict['language_model']['embedding'] = {'word_embeddings':{'weight':splited_word_embedding.clone()}}
            
        if is_last_stage:
            shard_encoder[get_weight_name('final_norm.weight')] = encoder[get_weight_name('final_norm.weight')].clone()
            shard_encoder[get_weight_name('final_norm.bias')] = encoder[get_weight_name('final_norm.bias')].clone()

            if untie_embeddings_and_output_weights:
                output_layer = whole_state_dict['language_model']['output_layer']['weight']
                splited_output_layer = output_layer.view(save_tensor_model_parallel_size, -1 ,output_layer.shape[1])[tensor_rank,:,:]
                splited_state_dict['language_model']['output_layer'] = {'weight':splited_output_layer.clone()}
            elif not is_first_stage:
                output_layer = whole_state_dict['language_model']['embedding']['word_embeddings']['weight']
                splited_output_layer = output_layer.view(save_tensor_model_parallel_size, -1 ,output_layer.shape[1])[tensor_rank,:,:]
                splited_state_dict['word_embeddings_for_head'] = {'weight':splited_output_layer.clone()}
            

        splited_state_dict['language_model']['encoder'] = shard_encoder
        
        print_with_time(f"{tensor_rank},{pipeline_rank}, {splited_state_dict['language_model']['encoder'].keys()}")
        for key, value in splited_state_dict['language_model']['encoder'].items():
            print_with_time(f"{key}, {value.shape}")
            # print(f"================= {torch.isnan(value).any() or torch.isinf(value).any()} ======================")
        if is_first_stage:
            print_with_time("> is first stage")
            for key, value in splited_state_dict['language_model']['embedding']['word_embeddings'].items():
                print_with_time(f"{key}, {value.shape}")
                # print(f"================= {torch.isnan(value).any() or torch.isinf(value).any()} ======================")
        if is_last_stage:
            print_with_time("> is last stage")
            if untie_embeddings_and_output_weights:
                for key, value in splited_state_dict['language_model']['output_layer'].items():
                    print_with_time(f"{key}, {value.shape}")
            elif not is_first_stage:
                for key, value in splited_state_dict['word_embeddings_for_head'].items():
                    print_with_time(f"{key}, {value.shape}")

                # print(f"================= {torch.isnan(value).any() or torch.isinf(value).any()}  ======================")
        
        
        return splited_state_dict

            
        
    directory_name = 'release'
    new_model_base = output_dir
    assert num_layers == (num_layers // save_pipeline_model_parallel_size) * save_pipeline_model_parallel_size, \
            "n_layers should be divisible by pipeline_model_parallel_size"
    layer_per_partition = num_layers // save_pipeline_model_parallel_size
    
    # pad vocab
    if untie_embeddings_and_output_weights:
        word_embedding, output_layer = pad_vocab(
            whole_state_dict['language_model']['embedding']['word_embeddings']['weight'], 
            whole_state_dict['language_model']['output_layer']['weight'], 
            hidden_size, vocab_size, make_vocab_size_divisible_by, save_tensor_model_parallel_size)
        whole_state_dict['language_model']['embedding']['word_embeddings']['weight'] = word_embedding
        whole_state_dict['language_model']['output_layer']['weight'] = output_layer

    else:
        word_embedding, output_layer = pad_vocab(
            whole_state_dict['language_model']['embedding']['word_embeddings']['weight'], 
            None, 
            hidden_size, vocab_size, make_vocab_size_divisible_by, save_tensor_model_parallel_size)
        
        whole_state_dict['language_model']['embedding']['word_embeddings']['weight'] = word_embedding
        


    pipeline_parallel = True if save_pipeline_model_parallel_size > 1 else False
    
    for tensor_rank in range(save_tensor_model_parallel_size):
        for pipeline_rank in range(save_pipeline_model_parallel_size):
            splited_state_dict = split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank)
            check_keys(splited_state_dict)
            
            state_dict = {
                'args': None,
                'checkpoint_version': 3.0,
                'iteration': 0,
                'model':splited_state_dict,
                'rng_state': None
            }

            # save checkpoint    
            if not pipeline_parallel:
                common_path = os.path.join(new_model_base, directory_name,
                                f'mp_rank_{tensor_rank:02d}')
            else:
                common_path = os.path.join(new_model_base, directory_name,
                                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
                
            os.makedirs(common_path, exist_ok=True)
            
            # distributed optimizer is not supported yet
            save_path = os.path.join(common_path, "model_optim_rng.pt")
            print(f'saving {save_path}')
            torch.save(state_dict, save_path)
    
    tracker_filename = os.path.join(output_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'w') as f:
        f.write('release')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    parser.add_argument("--input-dir", default=None,
                       help="Location of megatron checkpoint, which contains "
                       "a latest_checkpointed_iteration.txt file and a checkpoint dir.")
    parser.add_argument("--output-dir", default=None,
                       help="Location to save megatron checkpoint")
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    parser.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    parser.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    parser.add_argument('--gate-gelu', action='store_true',
                       help='Use gate gelu.')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='Vocabulary size.')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    parser.add_argument("--load-pipeline-model-parallel-size", default="1", type=int,
                        help="Number of pipeline model parallel size from checkpoint.")
    parser.add_argument("--load-tensor-model-parallel-size", default="1", type=int,
                        help="Number of tensor model parallel size from checkpoint.")
    parser.add_argument("--save-pipeline-model-parallel-size", default="1", type=int,
                        help="Number of pipeline model parallel size to save.")
    parser.add_argument("--save-tensor-model-parallel-size", default="1", type=int,
                        help="Number of tensor model parallel size to save.")
    
    args = parser.parse_args()
    
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    
    # load megatron checkpoint with model parallel
    whole_state_dict = load_checkpoint(
        args.input_dir, args.num_layers, args.hidden_size, args.add_bias_linear, 
        args.untie_embeddings_and_output_weights, args.gate_gelu,
        args.load_pipeline_model_parallel_size, args.load_tensor_model_parallel_size)
    
    # check_keys(whole_state_dict)
    
    # save megatron checkpoint with model parallel
    save_checkpoint(
        whole_state_dict, args.output_dir, args.num_layers, args.hidden_size, 
        args.num_attention_heads, args.add_bias_linear, args.untie_embeddings_and_output_weights, 
        args.gate_gelu, args.vocab_size, args.make_vocab_size_divisible_by,
        args.save_pipeline_model_parallel_size, args.save_tensor_model_parallel_size)
    
if __name__ == "__main__":
    # TODO: Support more activation such as swiglu 
    main()

    