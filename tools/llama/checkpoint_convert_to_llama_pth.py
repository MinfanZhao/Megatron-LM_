import argparse
import os
import torch
import sys
import gc


INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}
NUM_LAYERS = {
    "7B": 32,
    "13B": 40,
    "30B": 60,
    "65B": 80,
}
N_HEADS = {
    "7B": 32,
    "13B": 40,
    "30B": 52,
    "65B": 64,
}
DIM = {
    "7B": 4096,
    "13B": 5120,
    "30B": 6656,
    "65B": 8192, 
}



def combine_from_split_weight(megatron_model_base, model_size, tensor_parallel_size, pipeline_parallel_size):
    n_layers = NUM_LAYERS[model_size]
    n_heads = N_HEADS[model_size]
    dim = DIM[model_size]
    num_shards = tensor_parallel_size
    n_heads_per_shard = n_heads // num_shards
    
    dims_per_head = dim // n_heads
    pipeline_parallel = True if pipeline_parallel_size > 1 else False
    tensor_parallel = True if tensor_parallel_size > 1 else False
    assert n_layers == (n_layers // pipeline_parallel_size) * pipeline_parallel_size, "n_layers should be divisible by pipeline_size"
    layer_per_partition = n_layers // pipeline_parallel_size
    state_dict_shards = []
    
    tracker_filename = os.path.join(megatron_model_base, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'r') as f:
        iteration = f.read().strip()
        if iteration != 'release':
            iteration = int(iteration)
            megatron_model_base = os.path.join(
                megatron_model_base, 'iter_{:07d}'.format(iteration))
        else:
            megatron_model_base = os.path.join(
                megatron_model_base, 'release')
    
    
    for pipeline_rank in range(pipeline_parallel_size):
        state_dict_shards.append([])
        
        for tensor_rank in range(tensor_parallel_size):
            if not pipeline_parallel:
                common_path = os.path.join(megatron_model_base, 
                                f'mp_rank_{tensor_rank:02d}')
            else:
                common_path = os.path.join(megatron_model_base, 
                                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
            model_checkpoint_name = os.path.join(common_path, 'model_optim_rng.pt')
            print(f'loading {model_checkpoint_name}')
            state_dict_shard = torch.load(model_checkpoint_name, map_location='cpu')['model']['language_model']
            state_dict_shards[pipeline_rank].append(state_dict_shard)
            print(f'{model_checkpoint_name} loaded')


    megatron_state_dict = {'embedding': {'word_embeddings':{'weight':None}}, 
                            'encoder': {}, 
                            'output_layer': {'weight':None}}
    
    for pipeline_rank in range(pipeline_parallel_size):
        is_first_stage = True if pipeline_rank == 0 else False
        is_last_stage = True if pipeline_rank == pipeline_parallel_size - 1 else False
        layer_offset = layer_per_partition * pipeline_rank
        for layer_index in range(layer_per_partition):
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.input_rmsnorm.weight'] = \
                        state_dict_shards[pipeline_rank][0]['encoder'][f'layers.{layer_index}.input_rmsnorm.weight']
            
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.self_attention.query_key_value.weight'] = \
                torch.cat([tensor_shard['encoder'][f'layers.{layer_index}.self_attention.query_key_value.weight'].view(
                    1, -1, dim) for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(
                    -1, dim)
            # dense_shape = state_dict_shards[pipeline_rank][0]['encoder'][f'layers.{layer_offset + layer_index}.self_attention.dense.weight'].shape
            # print(f"====== attention qkv shape:{dense_shape}==========") 
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.self_attention.dense.weight'] = \
                torch.cat([tensor_shard['encoder'][f'layers.{layer_index}.self_attention.dense.weight'].view(
                    dim, 1, -1) for tensor_shard in state_dict_shards[pipeline_rank]], dim=1).view(
                    dim, dim)
                
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.post_attention_rmsnorm.weight'] = \
                state_dict_shards[pipeline_rank][0]['encoder'][f'layers.{layer_index}.post_attention_rmsnorm.weight']
                
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.mlp.dense_h_to_4h.weight'] = \
                torch.cat([tensor_shard['encoder'][f'layers.{layer_index}.mlp.dense_h_to_4h.weight'].view(
                    2, -1, dim) for tensor_shard in state_dict_shards[pipeline_rank]], dim=1).view(-1, dim)
            
            megatron_state_dict['encoder'][f'layers.{layer_offset + layer_index}.mlp.dense_4h_to_h.weight'] = \
                torch.cat([tensor_shard['encoder'][f'layers.{layer_index}.mlp.dense_4h_to_h.weight'] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=1).view(dim, -1)
            
        if is_first_stage:
            megatron_state_dict['embedding']['word_embeddings']['weight'] = \
                torch.cat([tensor_shard['embedding']['word_embeddings']['weight'] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
        
        if is_last_stage:
            megatron_state_dict['encoder']['final_rmsnorm.weight'] = \
                state_dict_shards[pipeline_rank][0]['encoder']['final_rmsnorm.weight']
            megatron_state_dict['output_layer']['weight'] = \
                torch.cat([tensor_shard['output_layer']['weight'] \
                    for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
                
    print("======== encoder =========")
    for key in megatron_state_dict['encoder'].keys():
        print(key, megatron_state_dict['encoder'][key].shape)
    print("======== embedding.word_embeddings =========")
    for key, value in megatron_state_dict['embedding']['word_embeddings'].items():
        print(key, value.shape)
    print("======== output_layer =========")
    for key, value in megatron_state_dict['output_layer'].items():
        print(key, value.shape)
            
    return megatron_state_dict

def convert_to_llama_state_dict(megatron_state_dict, model_size):
    n_layers = NUM_LAYERS[model_size]
    n_heads = N_HEADS[model_size]
    dim = DIM[model_size]
    num_shards = NUM_SHARDS[model_size]
    n_heads_per_shard = n_heads // num_shards
    dims_per_head = dim // n_heads
    llama_state_dict = {}
    rope_freqs = 1.0 / (10000 ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    
    for layer_i in range(n_layers):
        llama_state_dict[f"layers.{layer_i}.attention_norm.weight"] = \
            megatron_state_dict['encoder'][f"layers.{layer_i}.input_rmsnorm.weight"]
        wq,wk,wv = fix_attention_weight_to_llama(
            megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.query_key_value.weight"],
            n_heads, dims_per_head, dim)
        llama_state_dict[f"layers.{layer_i}.attention.wq.weight"] = wq.reshape(dim, dim)
        llama_state_dict[f"layers.{layer_i}.attention.wk.weight"] = wk.reshape(dim, dim)
        llama_state_dict[f"layers.{layer_i}.attention.wv.weight"] = wv.reshape(dim, dim)
        llama_state_dict[f"layers.{layer_i}.attention.wo.weight"] = \
            megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.dense.weight"]
        llama_state_dict[f"layers.{layer_i}.ffn_norm.weight"] = \
            megatron_state_dict['encoder'][f"layers.{layer_i}.post_attention_rmsnorm.weight"]
        w1,w3 = fix_swiglu_weight_to_llama(
            megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_h_to_4h.weight"])
        llama_state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = w1.reshape(-1, dim)
        llama_state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = \
            megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"]
        llama_state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = w3.reshape(-1, dim)
        llama_state_dict[f"layers.{layer_i}.attention.inner_attention.rope.freqs"] = rope_freqs.half()
            
        
    llama_state_dict[f"norm.weight"] = megatron_state_dict['encoder']["final_rmsnorm.weight"]
    llama_state_dict["tok_embeddings.weight"] = \
        megatron_state_dict['embedding']['word_embeddings']['weight']
    llama_state_dict["output.weight"] = \
        megatron_state_dict['output_layer']['weight']
        
    print("======= llama state dict =======")
    for key, value in llama_state_dict.items():
        print(key, value.shape)
    return llama_state_dict
        
    

def split_for_shards(llama_state_dict, num_shards):
    new_state_dicts = [dict() for _ in range(num_shards)]
    for k in list(llama_state_dict.keys()):
        v = llama_state_dict[k]
        if k is not None:
            if k=='tok_embeddings.weight':
                print(f"Processing {k}")
                assert v.size(1)%num_shards==0
                splits = v.split(v.size(1)//num_shards,dim=1)
            elif k=='output.weight':
                print(f"Processing {k}")
                if v.size(0)%num_shards==0:
                    splits = v.split(v.size(0)//num_shards,dim=0)
                else:
                    size_list = [v.size(0)//num_shards] * num_shards
                    size_list[-1] += v.size(0)%num_shards
                    splits = v.split(size_list, dim=0) # 13B: size_list == [24976,24977]
            elif k=='norm.weight':
                print(f"Processing {k}")
                splits = [v] * num_shards
            elif 'ffn_norm.weight' in k:
                print(f"Processing {k}")
                splits = [v] * num_shards
            elif 'attention_norm.weight' in k:
                print(f"Processing {k}")
                splits = [v] * num_shards


            elif 'w1.weight' in k:
                print(f"Processing {k}")
                splits = v.split(v.size(0)//num_shards,dim=0)
            elif 'w2.weight' in k:
                print(f"Processing {k}")
                splits = v.split(v.size(1)//num_shards,dim=1)
            elif 'w3.weight' in k:
                print(f"Processing {k}")
                splits = v.split(v.size(0)//num_shards,dim=0)


            elif 'wo.weight' in k:
                print(f"Processing {k}")
                splits = v.split(v.size(1)//num_shards,dim=1)

            elif 'wv.weight' in k:
                print(f"Processing {k}")
                splits = v.split(v.size(0)//num_shards,dim=0)

            elif "wq.weight" in k or "wk.weight" in k:
                print(f"Processing {k}")
                # v = unpermute(v)
                splits = v.split(v.size(0)//num_shards,dim=0)
            elif "rope.freqs" in k:
                print(f"Processing {k}")
                splits = [v] * num_shards
            else:
                print(f"Unexpected key {k}")
                raise ValueError
            for sd,split in zip(new_state_dicts,splits):
                sd[k] = split.clone()
                del split
            del splits
        del llama_state_dict[k],v
        gc.collect()    # Effectively enforce garbage collection
    return new_state_dicts
    

            
def fix_attention_weight_to_llama(query_key_value_weight, n_heads, dims_per_head, dim):
    wq,wk,wv = torch.unbind(query_key_value_weight.view(n_heads, 3, dims_per_head, dim), dim = 1)
    return wq,wk,wv


def fix_swiglu_weight_to_llama(swiglu_weight):
    weight_shape = swiglu_weight.shape
    w1,w3 = torch.unbind(swiglu_weight.view(2, -1, *weight_shape[2:]))
    return w1,w3



def convert_model(new_model_base, megatron_model_base, model_size, tensor_parallel_size, pipeline_parallel_size):
    # os.makedirs(new_model_base, exist_ok=True)
    megatron_state_dict = combine_from_split_weight(megatron_model_base, model_size, tensor_parallel_size, pipeline_parallel_size)
    llama_state_dict = convert_to_llama_state_dict(megatron_state_dict, model_size)
    new_model_base = os.path.join(new_model_base, model_size)
    os.makedirs(new_model_base, exist_ok=True)
    print(f"make dir path:{new_model_base}")
    if model_size == '7B':
        torch.save(llama_state_dict, os.path.join(new_model_base, 'consolidated.00.pth'))
    else:
        splits = split_for_shards(llama_state_dict, NUM_SHARDS[model_size])
        for index, shard in enumerate(splits):
            torch.save(shard, os.path.join(new_model_base, f'consolidated.0{index}.pth'))
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--megatron_path', type=str, default=None,
                       help='Base directory of Megatron repository')
    parser.add_argument(
        "--input_dir", default="./checkpoints/checkpoint-llama",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir", default="./checkpoints/llama-7B-p1t4/",
        help="Location to write llama version model and tokenizer",
    )
    # parser.add_argument(
    #     "--tokenizer_path", default="./checkpoints/checkpoint-llama",
    #     help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    # )
    parser.add_argument(
        "--model_size", default="7B",
        choices=['7B', '13B', '30B', '65B'],
    )
    parser.add_argument(
        "--pipeline_size", default="1", type=int
    )
    parser.add_argument(
        "--tensor_size", default="4", type=int
    )
    args = parser.parse_args()
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
        
    convert_model(
        new_model_base=args.output_dir,
        megatron_model_base=args.input_dir,
        model_size=args.model_size,
        tensor_parallel_size=args.tensor_size,
        pipeline_parallel_size=args.pipeline_size,
    )
    
    
if __name__ == "__main__":
    main()