# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import math
import os
import shutil
import warnings

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

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


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def convert_model(new_model_base, llama_model_path, model_size, tensor_size, pipeline_size):
    
    
    def read_state_dict(input_base_path):
        
        num_shards = NUM_SHARDS[model_size]

        print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
        # Load weights
        if model_size == "7B":
            # Not sharded
            # (The sharded implementation would also work, but this is simpler.)
            loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
        else:
            # Sharded
            loaded = [
                torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
                for i in range(num_shards)
            ]
        
        print("############## loaded ####################")
        if model_size == "7B":
            for key, value in loaded.items():
                print(key, value.shape)
        else:
            for shard_load in loaded:
                for key, value in shard_load.items():
                    print(key, value.shape)
        
        return loaded
    

    def convert_to_megatron_state_dict(llama_state_dict):
        
        num_shards = NUM_SHARDS[model_size]
        n_layers = NUM_LAYERS[model_size]
        n_heads = N_HEADS[model_size]
        n_heads_per_shard = n_heads // num_shards
        dim = DIM[model_size]
        dims_per_head = dim // n_heads
    
        megatron_state_dict = {'embedding': {'word_embeddings':{'weight':None}}, 
                               'encoder': {}, 
                               'output_layer': {'weight':None}}
        
        for layer_i in range(n_layers):
            if model_size == "7B":
                # print(f'{layer_i} input_rms : {llama_state_dict[f"layers.{layer_i}.attention_norm.weight"][:20]}')
                # print(f'{layer_i} wq : {llama_state_dict[f"layers.{layer_i}.attention.wq.weight"][0,:20]}')
                # print(f'{layer_i} wk : {llama_state_dict[f"layers.{layer_i}.attention.wk.weight"][0,:20]}')
                # print(f'{layer_i} wv : {llama_state_dict[f"layers.{layer_i}.attention.wv.weight"][0,:20]}')
                # print(f'{layer_i} wo : {llama_state_dict[f"layers.{layer_i}.attention.wo.weight"][0,:20]}')
                # print(f'{layer_i} post_attn_rms : {llama_state_dict[f"layers.{layer_i}.ffn_norm.weight"][:20]}')
                
                
                # print(f'{layer_i} w1 {llama_state_dict[f"layers.{layer_i}.feed_forward.w1.weight"][0,:20]}')
                # print(f'{layer_i} w3 {llama_state_dict[f"layers.{layer_i}.feed_forward.w3.weight"][0,:20]}')
                # print(f'{layer_i} w2 {llama_state_dict[f"layers.{layer_i}.feed_forward.w2.weight"][0,:20]}')
                
                # Unsharded
                megatron_state_dict['encoder'][f"layers.{layer_i}.input_rmsnorm.weight"] = \
                    llama_state_dict[f"layers.{layer_i}.attention_norm.weight"]
                megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.query_key_value.weight"] = \
                    fix_attention_weight(
                        llama_state_dict[f"layers.{layer_i}.attention.wq.weight"], 
                        llama_state_dict[f"layers.{layer_i}.attention.wk.weight"], 
                        llama_state_dict[f"layers.{layer_i}.attention.wv.weight"],
                        n_heads, dims_per_head, dim)
                megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.dense.weight"] = \
                    llama_state_dict[f"layers.{layer_i}.attention.wo.weight"]
                megatron_state_dict['encoder'][f"layers.{layer_i}.post_attention_rmsnorm.weight"] = \
                    llama_state_dict[f"layers.{layer_i}.ffn_norm.weight"]
                megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_h_to_4h.weight"] = \
                    fix_swiglu_weight(
                        llama_state_dict[f"layers.{layer_i}.feed_forward.w1.weight"],
                        llama_state_dict[f"layers.{layer_i}.feed_forward.w3.weight"]
                    )
                megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"] = \
                    llama_state_dict[f"layers.{layer_i}.feed_forward.w2.weight"]
    
                
            else:
                # Sharded
                # Note that in the 13B checkpoint, not cloning the two following weights will result in the checkpoint
                # becoming 37GB instead of 26GB for some reason.
                
                megatron_state_dict['encoder'][f"layers.{layer_i}.input_rmsnorm.weight"] = \
                    llama_state_dict[0][f"layers.{layer_i}.attention_norm.weight"]
                megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.query_key_value.weight"] = \
                    fix_attention_weight(
                        torch.cat(
                            [
                                llama_state_dict[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                                for i in range(num_shards)
                            ],
                            dim=0,
                        ).reshape(dim, dim), 
                        torch.cat(
                            [
                                llama_state_dict[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                                for i in range(num_shards)
                            ],
                            dim=0,
                        ).reshape(dim, dim), 
                        torch.cat(
                            [
                                llama_state_dict[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                                for i in range(num_shards)
                            ],
                            dim=0,
                        ).reshape(dim, dim),
                        n_heads, dims_per_head, dim)
                megatron_state_dict['encoder'][f"layers.{layer_i}.self_attention.dense.weight"] = torch.cat(
                    [llama_state_dict[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1)
                megatron_state_dict['encoder'][f"layers.{layer_i}.post_attention_rmsnorm.weight"] = \
                    llama_state_dict[0][f"layers.{layer_i}.ffn_norm.weight"]
                megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_h_to_4h.weight"] = \
                    fix_swiglu_weight(
                        torch.cat(
                            [llama_state_dict[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0),
                        torch.cat(
                            [llama_state_dict[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0))
                megatron_state_dict['encoder'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"] = torch.cat(
                    [llama_state_dict[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1)
                
            
        if model_size == "7B":
        #     # Unsharded
            print(f'word_embeddings:{llama_state_dict["tok_embeddings.weight"][0,:20]}')
            print(f'output norm:{llama_state_dict["norm.weight"][:20]}')
            print(f'output_layer:{llama_state_dict["output.weight"][0,:20]}')
            megatron_state_dict['encoder']["final_rmsnorm.weight"] = llama_state_dict[f"norm.weight"]
            megatron_state_dict['embedding']['word_embeddings']['weight'] = llama_state_dict["tok_embeddings.weight"]
            megatron_state_dict['output_layer']['weight'] = llama_state_dict["output.weight"]
        else:
            megatron_state_dict['encoder']["final_rmsnorm.weight"] = llama_state_dict[0]["norm.weight"]
            megatron_state_dict['embedding']['word_embeddings']['weight'] = \
                torch.cat([llama_state_dict[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1)
            megatron_state_dict['output_layer']['weight'] = \
                torch.cat([llama_state_dict[i]["output.weight"] for i in range(num_shards)], dim=0)
            
            
            
        print("############### state dict encoder ###################")
        for key, value in megatron_state_dict['encoder'].items():
            print(key, value.shape)
        print("############### state dict embedding ###################")
        for key, value in megatron_state_dict['embedding']['word_embeddings'].items():
            print(key, value.shape)
        for key, value in megatron_state_dict['output_layer'].items():
            print(key, value.shape)
        print("############### end state dict ###################")
            
        return megatron_state_dict
    
    
    def split_and_save_for_parallel(new_model_base, whole_state_dict, tensor_size, pipeline_size):
        num_shards = NUM_SHARDS[model_size]
        n_layers = NUM_LAYERS[model_size]
        n_heads = N_HEADS[model_size]
        n_heads_per_shard = n_heads // num_shards
        dim = DIM[model_size]
        dims_per_head = dim // n_heads
        
        def split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank, tensor_size, pipeline_size):
            is_first_stage = True if pipeline_rank == 0 else False
            is_last_stage = True if pipeline_rank == pipeline_size - 1 else False
            splited_state_dict = {"encoder": {}}
            assert n_layers == (n_layers // pipeline_size) * pipeline_size, "n_layers should be divisible by pipeline_size"
            layer_per_partition = n_layers // pipeline_size
            start_layer_index = layer_per_partition * pipeline_rank
            
            for layer_index in range(layer_per_partition):
                splited_state_dict['encoder'][f"layers.{layer_index}.input_rmsnorm.weight"] = \
                    whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.input_rmsnorm.weight"]
                    
                attention_qkv = whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.self_attention.query_key_value.weight"]
                attention_qkv_shape = attention_qkv.shape
                splited_attention_qkv = attention_qkv.view(tensor_size, -1, attention_qkv_shape[1])[tensor_rank,:,:]
                splited_attention_qkv = splited_attention_qkv.reshape(-1, attention_qkv_shape[1])
                splited_state_dict['encoder'][f"layers.{layer_index}.self_attention.query_key_value.weight"] = splited_attention_qkv
                    
                    
                attention_dense = whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.self_attention.dense.weight"]
                splited_attention_dense = attention_dense.view(attention_dense.shape[0],tensor_size, -1)[:,tensor_rank,:]
                splited_state_dict['encoder'][f"layers.{layer_index}.self_attention.dense.weight"] = splited_attention_dense
                    
                    
                splited_state_dict['encoder'][f"layers.{layer_index}.post_attention_rmsnorm.weight"] = \
                    whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.post_attention_rmsnorm.weight"]
                    
                    
                mlp_in = whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.mlp.dense_h_to_4h.weight"]
                splited_mlp_in = mlp_in.view(2, tensor_size, -1, mlp_in.shape[1])[:,tensor_rank,:,:]
                splited_mlp_in = splited_mlp_in.reshape(-1, mlp_in.shape[1])
                splited_state_dict['encoder'][f"layers.{layer_index}.mlp.dense_h_to_4h.weight"] = splited_mlp_in
                   
                    
                mlp_out = whole_state_dict['encoder'][f"layers.{layer_index + start_layer_index}.mlp.dense_4h_to_h.weight"]
                splited_mlp_out = mlp_out.view(mlp_out.shape[0], tensor_size, -1)[:,tensor_rank,:]
                splited_state_dict['encoder'][f"layers.{layer_index}.mlp.dense_4h_to_h.weight"] = splited_mlp_out
                    
            if is_first_stage:
                word_embeddings = whole_state_dict['embedding']['word_embeddings']['weight']
                splited_word_embedding = word_embeddings.view(tensor_size, -1 ,word_embeddings.shape[1])[tensor_rank,:,:]
                splited_state_dict['embedding'] = {'word_embeddings':{'weight':splited_word_embedding}}
                
            if is_last_stage:
                output_layer = whole_state_dict['output_layer']['weight']
                splited_output_layer = output_layer.view(tensor_size, -1 ,output_layer.shape[1])[tensor_rank,:,:]
                splited_state_dict['output_layer'] = {'weight':splited_output_layer}
                # print('show keys:')
                # print(whole_state_dict['encoder'].keys())
                # print(splited_state_dict['encoder'].keys())
                splited_state_dict['encoder']["final_rmsnorm.weight"] = whole_state_dict['encoder']["final_rmsnorm.weight"]
            
            # print(tensor_rank, pipeline_rank, splited_state_dict['encoder'].keys())
            for key in splited_state_dict['encoder'].keys():
                print(key, splited_state_dict['encoder'][key].shape)
            if is_first_stage:
                for key, value in splited_state_dict['embedding']['word_embeddings'].items():
                    print(key, value.shape)
            if is_last_stage:
                for key, value in splited_state_dict['output_layer'].items():
                    print(key, value.shape)
            
            
            return splited_state_dict

            
        
        directory_name = 'release'
        
        pipeline_parallel = True if pipeline_size > 1 else False
        tensor_parallel = True if tensor_size > 1 else False
        
        for tensor_rank in range(tensor_size):
            for pipeline_rank in range(pipeline_size):
                splited_state_dict = split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank, tensor_size, pipeline_size)
                
                state_dict = {
                    'args': None,
                    'checkpoint_version': 3.0,
                    'iteration': 0,
                    'model':{'language_model': splited_state_dict},
                    'rng_state': None
                }

                # for save
        
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
    
      
    os.makedirs(new_model_base, exist_ok=True)
    # megatron_state_dict = torch.load(megatron_model_path, map_location="cpu")
    
    llama_state_dict = read_state_dict(llama_model_path)
    llama_to_megatron_state_dict = convert_to_megatron_state_dict(llama_state_dict)
    split_and_save_for_parallel(new_model_base, llama_to_megatron_state_dict, tensor_size, pipeline_size)    
    
    tracker_filename = os.path.join(new_model_base, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'w') as f:
        f.write('release')



def fix_attention_weight(wq, wk, wv, n_heads, dims_per_head, dim):
    query_key_value_weight = torch.cat([
        wq.view(n_heads,1,dims_per_head, dim),
        wk.view(n_heads,1,dims_per_head, dim),
        wv.view(n_heads,1,dims_per_head, dim)],dim=1).view(3*dim, dim)
    return query_key_value_weight

def fix_swiglu_weight(w1, w3):
    dense_h_to_4h_weight = torch.cat([w1,w3], dim=0)
    return dense_h_to_4h_weight



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="./checkpoints/checkpoint-llama",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    # parser.add_argument(
    #     "--megatron_model_path", default="./checkpoints/llama_mini/"
    # )
    parser.add_argument(
        "--model_size", default="7B",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir", default="./checkpoints/llama-7B-p1t4/",
        help="Location to write megatron version model and tokenizer",
    )
    parser.add_argument(
        "--pipeline_size", default="1", type=int
    )
    parser.add_argument(
        "--tensor_size", default="4", type=int
    )
    args = parser.parse_args()
    convert_model(
        new_model_base=args.output_dir,
        llama_model_path=os.path.join(args.input_dir, args.model_size),
        model_size=args.model_size,
        tensor_size=args.tensor_size,
        pipeline_size=args.pipeline_size,
    )


if __name__ == "__main__":
    main()