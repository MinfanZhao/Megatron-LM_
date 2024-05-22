import argparse
import json
import os
import shutil
import sys
import torch
from einops import rearrange

sys.path.insert(0, './')

def fix_qkv_order(weight, is_bias=False):
    if is_bias:
        print(f" === qkv bias shape:{weight.shape} ===")
        weight = weight.reshape(12, 3, 64).transpose(0,1).reshape(2304)
    else:
        print(f" === qkv proj shape:{weight.shape} ===")
        weight = weight.reshape(12, 3, 64, 768).transpose(0,1).reshape(2304, 768)
    
    return weight
    

def combine_from_split_weight(megatron_model_base, layers, tensor_num, pipeline_num, pipeline_split=None, specified_iter = None) :
    state_dict_shards = []
    not_pipeline_parallel = False if pipeline_num > 1 else True
    # if specified_iter is not None:
    megatron_model_base = os.path.join(megatron_model_base, 'iter_{:07d}'.format(specified_iter))
    # else:
    #     tracker_filename = os.path.join(megatron_model_base, 'latest_checkpointed_iteration.txt')
    #     with open(tracker_filename, 'r') as f:
    #         iteratrion = f.read().strip()
    #         if iteratrion != 'release':
    #             iteratrion = int(iteratrion)
    #             megatron_model_base = os.path.join(
    #                 megatron_model_base, 'iter_{:07d}'.format(iteratrion)
    #             )
    #         else:
    #             megatron_model_base = os.path.join(
    #                 megatron_model_base, 'release'
    #             )
    
    for pipeline_rank in range(pipeline_num):
        state_dict_shards.append([])
        for tensor_rank in range(tensor_num):
            if not_pipeline_parallel:
                common_path = os.path.join(megatron_model_base, 
                                           f'mp_rank_{tensor_rank:02d}')
            else:
                common_path = os.path.join(megatron_model_base,
                                           f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
            model_checkpoint_name = os.path.join(common_path, 'model_optim_rng.pt')
            print(f'loading {model_checkpoint_name}')
            state_dict_shard = torch.load(model_checkpoint_name, map_location='cpu')['model']
            state_dict_shards[pipeline_rank].append(state_dict_shard)
            print(f'{model_checkpoint_name} loaded')
    
    megatron_state_dict = {}

    for pipeline_rank in range(pipeline_num):
        if pipeline_split is None:
            assert layers == (layers // pipeline_num) * pipeline_num, 'layer_num should be divisible by pipeline num'
            layer_per_partition = layers // pipeline_num
            layer_offset = layer_per_partition * pipeline_rank
        else:
            layer_per_partition = pipeline_split[pipeline_rank]
            layer_offset = sum(pipeline_split[:pipeline_rank])
        is_first_stage = True if pipeline_rank == 0 else False
        is_last_stage = True if pipeline_rank == pipeline_num - 1 else False
        for layer_index in range(layer_per_partition):
            
            if (layer_index + layer_offset) % 2 == 1:
                megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.attn_mask'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.attn_mask']
            

            # Input LayerNorm Weights
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.input_layernorm.weight'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.input_layernorm.weight']

            # Input LayerNorm Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.input_layernorm.bias'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.input_layernorm.bias']

            # Self-Attention Query, Key, Value Weights
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.query_key_value.weight'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.self_attention.query_key_value.weight'].view(1,-1,2304) \
                           for tensor_shard in state_dict_shards[pipeline_rank]], dim=0).view(-1, 2304)

            # Self-Attention Query, Key, Value Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.query_key_value.bias'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.self_attention.query_key_value.bias'] \
                           for tensor_shard in state_dict_shards[pipeline_rank]], dim=0)
            
            # Self-Attention RPE
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.relative_position_bias_table'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.self_attention.relative_position_bias_table']\
                          for tensor_shard in state_dict_shards[pipeline_rank]], 1)
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.relative_position_index'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.self_attention.relative_position_index']

            # Self-Attention Dense Weight
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.dense.weight'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.self_attention.dense.weight'] \
                           for tensor_shard in state_dict_shards[pipeline_rank]], 1)

            # Self-Attention Dense Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.self_attention.dense.bias'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.self_attention.dense.bias']

            # Post-Attention LayerNorm Weight
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.post_attention_layernorm.weight'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.post_attention_layernorm.weight']

            # Post-Attention LayerNorm Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.post_attention_layernorm.bias'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.post_attention_layernorm.bias']

            # MLP Dense h-to-4h Weight
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.mlp.dense_h_to_4h.weight'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.mlp.dense_h_to_4h.weight'].view(1,-1,768) \
                           for tensor_shard in state_dict_shards[pipeline_rank]], 0).view(-1,768)

            # MLP Dense h-to-4h Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.mlp.dense_h_to_4h.bias'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.mlp.dense_h_to_4h.bias'] \
                           for tensor_shard in state_dict_shards[pipeline_rank]])

            # MLP Dense 4h-to-h Weight
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.mlp.dense_4h_to_h.weight'] = \
                torch.cat([tensor_shard[f'language_model.transformer.layers.{layer_index}.mlp.dense_4h_to_h.weight'] \
                           for tensor_shard in state_dict_shards[pipeline_rank]],1)

            # MLP Dense 4h-to-h Bias
            megatron_state_dict[f'language_model.transformer.layers.{layer_index + layer_offset}.mlp.dense_4h_to_h.bias'] = \
                state_dict_shards[pipeline_rank][0][f'language_model.transformer.layers.{layer_index}.mlp.dense_4h_to_h.bias']



        if is_first_stage:
            megatron_state_dict['language_model.proj.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.proj.weight']
            megatron_state_dict['language_model.proj.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.proj.bias']
            megatron_state_dict['language_model.proj_bulk.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.proj_bulk.weight']
            megatron_state_dict['language_model.proj_bulk.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.proj_bulk.bias']

            megatron_state_dict['language_model.down_blk.conv.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.down_blk.conv.weight']
            megatron_state_dict['language_model.down_blk.conv.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.down_blk.conv.bias']

            for i in [0,1,3,4]:
                megatron_state_dict[f'language_model.down_blk.b.{i}.weight'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.down_blk.b.{i}.weight']
                megatron_state_dict[f'language_model.down_blk.b.{i}.bias'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.down_blk.b.{i}.bias']

            megatron_state_dict['language_model.down_blk2.conv.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.down_blk2.conv.weight']
            megatron_state_dict['language_model.down_blk2.conv.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.down_blk2.conv.bias']

            for i in [0,1,3,4]:
                megatron_state_dict[f'language_model.down_blk2.b.{i}.weight'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.down_blk2.b.{i}.weight']
                megatron_state_dict[f'language_model.down_blk2.b.{i}.bias'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.down_blk2.b.{i}.bias']

        if is_last_stage:
            print(f"pp rank:{pipeline_rank} {state_dict_shards[pipeline_rank][0].keys()}")
            megatron_state_dict['language_model.up_blk.conv.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.up_blk.conv.weight']
            megatron_state_dict['language_model.up_blk.conv.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.up_blk.conv.bias']

            for i in [0,1,3,4]:
                megatron_state_dict[f'language_model.up_blk.b.{i}.weight'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.up_blk.b.{i}.weight']
                megatron_state_dict[f'language_model.up_blk.b.{i}.bias'] = \
                    state_dict_shards[pipeline_rank][0][f'language_model.up_blk.b.{i}.bias']

            megatron_state_dict['language_model.fc.weight'] = \
                state_dict_shards[pipeline_rank][0]['language_model.fc.weight']
            megatron_state_dict['language_model.fc.bias'] = \
                state_dict_shards[pipeline_rank][0]['language_model.fc.bias']
    
    return megatron_state_dict


def convert_to_pth_state_dict(megatron_state_dict, layers):
    pth_state_dict = {}

    pth_state_dict['module.proj.weight'] = \
        megatron_state_dict['language_model.proj.weight']
    pth_state_dict['module.proj.bias'] = \
        megatron_state_dict['language_model.proj.bias']
    pth_state_dict['module.proj_bulk.weight'] = \
        megatron_state_dict['language_model.proj_bulk.weight']
    pth_state_dict['module.proj_bulk.bias'] = \
        megatron_state_dict['language_model.proj_bulk.bias']
    
    pth_state_dict['module.down_blk.conv.weight'] = \
        megatron_state_dict['language_model.down_blk.conv.weight']
    pth_state_dict['module.down_blk.conv.bias'] = \
        megatron_state_dict['language_model.down_blk.conv.bias']
    
    for i in [0,1,3,4]:
        pth_state_dict[f'module.down_blk.b.{i}.weight'] = \
            megatron_state_dict[f'language_model.down_blk.b.{i}.weight']
        pth_state_dict[f'module.down_blk.b.{i}.bias'] = \
            megatron_state_dict[f'language_model.down_blk.b.{i}.bias']
    
    pth_state_dict['module.down_blk2.conv.weight'] = \
        megatron_state_dict['language_model.down_blk2.conv.weight']
    pth_state_dict['module.down_blk2.conv.bias'] = \
        megatron_state_dict['language_model.down_blk2.conv.bias']
    
    for i in [0,1,3,4]:
        pth_state_dict[f'module.down_blk2.b.{i}.weight'] = \
            megatron_state_dict[f'language_model.down_blk2.b.{i}.weight']
        pth_state_dict[f'module.down_blk2.b.{i}.bias'] = \
            megatron_state_dict[f'language_model.down_blk2.b.{i}.bias']
        
    for layer_i in range(layers):
        if layer_i % 2 == 1:
            pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn_mask'] = \
                megatron_state_dict[f'language_model.transformer.layers.{layer_i}.attn_mask']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.norm1.weight'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.input_layernorm.weight']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.norm1.bias'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.input_layernorm.bias']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.qkv.weight'] = \
            fix_qkv_order(megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.query_key_value.weight'])
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.qkv.bias'] = \
            fix_qkv_order(megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.query_key_value.bias'], is_bias=True)
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.relative_position_bias_table'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.relative_position_bias_table']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.relative_position_index'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.relative_position_index']  
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.proj.weight'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.dense.weight']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.attn.proj.bias'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.dense.bias']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.norm2.weight'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.post_attention_layernorm.weight']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.norm2.bias'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.post_attention_layernorm.bias']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc1.weight'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_h_to_4h.weight']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc1.bias'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_h_to_4h.bias']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc2.weight'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_4h_to_h.weight']
        pth_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc2.bias'] = \
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_4h_to_h.bias']
    
    pth_state_dict['module.up_blk.conv.weight'] = \
        megatron_state_dict['language_model.up_blk.conv.weight']
    pth_state_dict['module.up_blk.conv.bias'] = \
        megatron_state_dict['language_model.up_blk.conv.bias']
    
    for i in [0,1,3,4]:
        pth_state_dict[f'module.up_blk.b.{i}.weight'] = \
            megatron_state_dict[f'language_model.up_blk.b.{i}.weight']
        pth_state_dict[f'module.up_blk.b.{i}.bias'] = \
            megatron_state_dict[f'language_model.up_blk.b.{i}.bias']
    
    pth_state_dict['module.fc.weight'] = \
        megatron_state_dict['language_model.fc.weight']
    pth_state_dict['module.fc.bias'] = \
        megatron_state_dict['language_model.fc.bias']

    print("======= pth state dict =======")
    for key, value in pth_state_dict.items():
        print(key, value.shape)
    pth_dict = {
        'model_state_dict': pth_state_dict,
        'epoch': 50,
    }
    return pth_dict


def convert_model(new_model_path, megatron_model_base, layers, tp_size, pp_size, pipeline_split=None, specified_iter=None):
    if pipeline_split is not None:
        pp_size = len(pipeline_split)
    megatron_state_dict = combine_from_split_weight(megatron_model_base, layers, tp_size, pp_size, pipeline_split, specified_iter=specified_iter)
    # print(megatron_state_dict)
    pth_state_dict = convert_to_pth_state_dict(megatron_state_dict, layers)
    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path, exist_ok=True)
    print(f"make dir path: {new_model_path}")
    save_path = os.path.join(new_model_path, f'iter_{specified_iter:07d}.pth')
    print(f"save checkpoint at: {save_path}")
    torch.save(pth_state_dict, save_path)
    print("done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="./checkpoint/wenhai-test",
        help="Base directory of Megatron reposityory",
    )
    parser.add_argument(
        "--layers", default="10", type=int,
        choices=["10", "12"],
    )
    parser.add_argument(
        "--output-dir", default="./checkpoint/wenhai_to_torch_pth/finetune_10_days_bs16",
        help="Location to write torch version model",
    )
    parser.add_argument(
        "--pipeline-num", default="2", type=int
    )
    parser.add_argument(
        "--tensor-num", default="1", type=int
    )
    parser.add_argument('--inbalance-pipeline-stage', '-p', type=int, nargs='+')
    parser.add_argument(
        "--specified-iter", '-i', default=150, type=int, required=True
    )
    
    args = parser.parse_args()
    print(f"{args.inbalance_pipeline_stage}")
    convert_model(
        new_model_path=args.output_dir,
        megatron_model_base=args.input_dir,
        layers=args.layers,
        tp_size=args.tensor_num,
        pp_size=args.pipeline_num,
        pipeline_split=args.inbalance_pipeline_stage,
        specified_iter=args.specified_iter
    )


if __name__ == "__main__":
    main()