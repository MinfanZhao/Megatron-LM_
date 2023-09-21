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
import sys
sys.path.append('./')

from torch.nn.parameter import Parameter
from megatron.model.utils import init_method_normal, scaled_init_method_normal
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


def combine_model(new_model_base, llama_model_path, model_size, 
                  target_vocab_size, init_method_std, dtype=torch.float16):
    
    
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
    

    def replace_word_embedding(llama_state_dict, target_vocab_size):
        
        num_shards = NUM_SHARDS[model_size]
        n_layers = NUM_LAYERS[model_size]
        n_heads = N_HEADS[model_size]
        n_heads_per_shard = n_heads // num_shards
        dim = DIM[model_size]
        dims_per_head = dim // n_heads
    
        
        # use Megatron-LM initialization
        word_embedding_weight = torch.empty(target_vocab_size, dim,
                                dtype=torch.float,
                                requires_grad=False)
        output_weight = torch.empty(target_vocab_size, dim,
                                dtype=torch.float,
                                requires_grad=False)
        init_method = init_method_normal(init_method_std)
        
        init_method(word_embedding_weight)
        init_method(output_weight)
        
        # convert data type
        initialized_embedding_weight = word_embedding_weight.to(dtype=dtype)
        initialized_output_weight = output_weight.to(dtype=dtype)

        if model_size == '7B':
            old_word_embedding_weight = llama_state_dict["tok_embeddings.weight"]
            old_output_weight = llama_state_dict["output.weight"]
        else:
            old_word_embedding_weight = torch.cat([shard["tok_embeddings.weight"] for shard in llama_state_dict], dim=1)
            old_output_weight = torch.cat([shard["output.weight"] for shard in llama_state_dict], dim=0)

        # check shape
        print(f"origin tok_embeddings weight shape f{old_word_embedding_weight.shape}")
        print(f"pad tok_embeddings weight shape f{initialized_embedding_weight[llama_state_dict['tok_embeddings.weight'].shape[0]:,:].shape}")
        
        print(f"origin output weight shape f{old_output_weight.shape}")
        print(f"pad output weight shape f{initialized_output_weight[llama_state_dict['output.weight'].shape[0]:,:].shape}")
        
        # expand word embedding and output weight
        new_word_embedding_weight = torch.cat([old_word_embedding_weight, 
                                              initialized_embedding_weight[llama_state_dict["tok_embeddings.weight"].shape[0]:,].clone()], dim=0)
        new_output_weight = torch.cat([old_output_weight, 
                                    initialized_output_weight[llama_state_dict["output.weight"].shape[0]:,:].clone()], dim=0)
        
        if model_size == '7B':
            llama_state_dict["tok_embeddings.weight"] = new_word_embedding_weight
            llama_state_dict["output.weight"] = new_output_weight
        else:
            word_embedding_weight_shard = torch.split(new_word_embedding_weight, num_shards, dim=1)
            output_weight_shard = torch.split(new_output_weight, num_shards, dim=0)
            for i in num_shards:
                llama_state_dict[i]["tok_embeddings.weight"] = word_embedding_weight_shard[i].clone()
                llama_state_dict[i]["output.weight"] = output_weight_shard[i].clone()
            
        return llama_state_dict
    
    new_model_base = os.path.join(new_model_base, model_size)
    os.makedirs(new_model_base, exist_ok=True)
    
    num_shards = NUM_SHARDS[model_size]
    llama_state_dict = read_state_dict(llama_model_path)
    new_llama_state_dict = replace_word_embedding(llama_state_dict, target_vocab_size)
    
    print("saving checkpoints...")
    if model_size == '7B':
        torch.save(new_llama_state_dict,os.path.join(new_model_base, f"consolidated.00.pth")) 
    else:
        for i in range(num_shards):
            torch.save(new_llama_state_dict[i],os.path.join(new_model_base, f"consolidated.{i:02d}.pth")) 

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="/acsa-med/megatron-llama/torch-pretrain/meta-llama",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size", default="7B",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--output_dir", default="/acsa-med/megatron-llama/torch-pretrain/extend-meta-llama",
        help="Location to write megatron version model and tokenizer",
    )
    parser.add_argument(
        "--target_vocab_size", default="49954", type=int
    )
    parser.add_argument(
        "--init-method-std", type=float, default="0.002"
    )
    args = parser.parse_args()
    combine_model(
        new_model_base=args.output_dir,
        llama_model_path=os.path.join(args.input_dir, args.model_size),
        model_size=args.model_size,
        target_vocab_size = args.target_vocab_size,
        init_method_std = args.init_method_std
    )


if __name__ == "__main__":
    main()