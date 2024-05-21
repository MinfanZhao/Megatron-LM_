import argparse
import json
import os
import shutil

import torch
from einops import rearrange

def fix_qkv_order(weight, is_bias=False):
    if is_bias:
        print(f" === qkv bias shape:{weight.shape} ===")
        weight = weight.reshape(3, 12, 64).transpose(0,1).reshape(2304)
    else:
        print(f" === qkv proj shape:{weight.shape} ===")
        weight = weight.reshape(3, 12, 64, 768).transpose(0,1).reshape(2304, 768)
    
    return weight
    


def convert_model(new_model_path, orig_model_path, layers, tensor_num, pipeline_num, pipeline_split=None):

    def read_state_dict(input_base_path):
        loaded = torch.load(input_base_path)['model_state_dict']
        print('Model loaded.')
        for key, value in loaded.items():
            print(key, value.shape)
        return loaded
    
    def convert_to_megatron_state_dict(orig_state_dict, tensor_size):
        megatron_state_dict = {}

        megatron_state_dict['language_model.proj.weight'] = \
            orig_state_dict['module.proj.weight']
        megatron_state_dict['language_model.proj.bias'] = \
            orig_state_dict['module.proj.bias']
        megatron_state_dict['language_model.proj_bulk.weight'] = \
            orig_state_dict['module.proj_bulk.weight']
        megatron_state_dict['language_model.proj_bulk.bias'] = \
            orig_state_dict['module.proj_bulk.bias']
        
        megatron_state_dict['language_model.down_blk.conv.weight'] = \
            orig_state_dict['module.down_blk.conv.weight']
        megatron_state_dict['language_model.down_blk.conv.bias'] = \
            orig_state_dict['module.down_blk.conv.bias']
        
        for i in [0,1,3,4]:
            megatron_state_dict[f'language_model.down_blk.b.{i}.weight'] = \
                orig_state_dict[f'module.down_blk.b.{i}.weight']
            megatron_state_dict[f'language_model.down_blk.b.{i}.bias'] = \
                orig_state_dict[f'module.down_blk.b.{i}.bias']
        
        megatron_state_dict['language_model.down_blk2.conv.weight'] = \
            orig_state_dict['module.down_blk2.conv.weight']
        megatron_state_dict['language_model.down_blk2.conv.bias'] = \
            orig_state_dict['module.down_blk2.conv.bias']
        
        for i in [0,1,3,4]:
            megatron_state_dict[f'language_model.down_blk2.b.{i}.weight'] = \
                orig_state_dict[f'module.down_blk2.b.{i}.weight']
            megatron_state_dict[f'language_model.down_blk2.b.{i}.bias'] = \
                orig_state_dict[f'module.down_blk2.b.{i}.bias']
            
        
        for layer_i in range(layers):
            if layer_i % 2 == 1:
                megatron_state_dict[f'language_model.transformer.layers.{layer_i}.attn_mask'] = \
                    orig_state_dict[f'module.layers.0.blocks.{layer_i}.attn_mask']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.input_layernorm.weight'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.norm1.weight']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.input_layernorm.bias'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.norm1.bias']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.query_key_value.weight'] = \
                fix_qkv_order(orig_state_dict[f'module.layers.0.blocks.{layer_i}.attn.qkv.weight'])
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.query_key_value.bias'] = \
                fix_qkv_order(orig_state_dict[f'module.layers.0.blocks.{layer_i}.attn.qkv.bias'], is_bias=True)
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.relative_position_bias_table'] = \
                orig_state_dict[f'module.layers.0.blocks.0.attn.relative_position_bias_table']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.relative_position_index'] = \
                orig_state_dict[f'module.layers.0.blocks.0.attn.relative_position_index']  
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.dense.weight'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.attn.proj.weight']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.self_attention.dense.bias'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.attn.proj.bias']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.post_attention_layernorm.weight'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.norm2.weight']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.post_attention_layernorm.bias'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.norm2.bias']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_h_to_4h.weight'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc1.weight']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_h_to_4h.bias'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc1.bias']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_4h_to_h.weight'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc2.weight']
            megatron_state_dict[f'language_model.transformer.layers.{layer_i}.mlp.dense_4h_to_h.bias'] = \
                orig_state_dict[f'module.layers.0.blocks.{layer_i}.mlp.fc2.bias']
        
        megatron_state_dict['language_model.up_blk.conv.weight'] = \
            orig_state_dict['module.up_blk.conv.weight']
        megatron_state_dict['language_model.up_blk.conv.bias'] = \
            orig_state_dict['module.up_blk.conv.bias']
        
        for i in [0,1,3,4]:
            megatron_state_dict[f'language_model.up_blk.b.{i}.weight'] = \
                orig_state_dict[f'module.up_blk.b.{i}.weight']
            megatron_state_dict[f'language_model.up_blk.b.{i}.bias'] = \
                orig_state_dict[f'module.up_blk.b.{i}.bias']
        
        megatron_state_dict['language_model.fc.weight'] = \
            orig_state_dict['module.fc.weight']
        megatron_state_dict['language_model.fc.bias'] = \
            orig_state_dict['module.fc.bias']
        
        print("############### state dict  ###################")
        for key, value in megatron_state_dict.items():
            print(key, value.shape)
        print("############### end state dict ###################")

        return megatron_state_dict
    
    def split_and_save_for_parallel(new_model_path, whole_state_dict, tensor_num, pipeline_num, n_layer):

        def split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank, tensor_num, pipeline_num, pipeline_split=None):
            is_first_stage = True if pipeline_rank == 0 else False
            is_last_stage = True if pipeline_rank == pipeline_num - 1 else False
            splited_state_dict = {}
            if pipeline_split is None:
                assert n_layer == (n_layer // pipeline_num) * pipeline_num, "n_layers should be divisible by pipeline_num"
                layer_per_partition = n_layer // pipeline_num
                start_layer_index = layer_per_partition * pipeline_rank
            else:
                layer_per_partition = pipeline_split[pipeline_rank]
                start_layer_index = sum(pipeline_split[:pipeline_rank])
            

            for layer_index in range(layer_per_partition):
                if (layer_index + start_layer_index) % 2 == 1:
                    # attn_mask = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.attn_mask']
                    # attn_mask_shape = attn_mask.shape
                    # splited_attn_mask = attn_mask.view(tensor_num, -1, attn_mask_shape[1])[tensor_rank,:,:]
                    splited_state_dict[f'language_model.transformer.layers.{layer_index}.attn_mask'] = \
                        whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.attn_mask']
                

                # Input LayerNorm Weights
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.input_layernorm.weight'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.input_layernorm.weight']

                # Input LayerNorm Bias
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.input_layernorm.bias'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.input_layernorm.bias']

                # Self-Attention Query, Key, Value Weights
                query_key_value_weight = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.query_key_value.weight']
                query_key_value_weight_shape = query_key_value_weight.shape
                splited_query_key_value_weight = query_key_value_weight.view(tensor_num, -1, query_key_value_weight_shape[1])[tensor_rank,:,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.query_key_value.weight'] = splited_query_key_value_weight.clone()

                # Self-Attention Query, Key, Value Bias
                query_key_value_bias = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.query_key_value.bias']
                splited_query_key_value_bias = query_key_value_bias.view(tensor_num, -1)[tensor_rank,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.query_key_value.bias'] = splited_query_key_value_bias.clone()
                
                # Self-Attention RPE
                rpe_shape = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.relative_position_bias_table'].shape
                
                splited_rpe_table_weight = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.relative_position_bias_table'].view(rpe_shape[0], tensor_num, -1)
                
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.relative_position_bias_table'] = splited_rpe_table_weight[:,tensor_rank].clone()
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.relative_position_index'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.relative_position_index']

                # Self-Attention Dense Weight
                dense_weight = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.dense.weight']
                dense_weight_shape = dense_weight.shape
                splited_dense_weight = dense_weight.view(dense_weight_shape[0], tensor_num, -1)[:,tensor_rank,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.dense.weight'] = splited_dense_weight.clone()

                # Self-Attention Dense Bias
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.self_attention.dense.bias'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.dense.bias']

                # Post-Attention LayerNorm Weight
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.post_attention_layernorm.weight'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.post_attention_layernorm.weight']

                # Post-Attention LayerNorm Bias
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.post_attention_layernorm.bias'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.post_attention_layernorm.bias']

                # MLP Dense h-to-4h Weight
                mlp_h_to_4h_weight = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.mlp.dense_h_to_4h.weight']
                mlp_h_to_4h_weight_shape = mlp_h_to_4h_weight.shape
                splited_mlp_h_to_4h_weight = mlp_h_to_4h_weight.view(tensor_num, -1, mlp_h_to_4h_weight_shape[1])[tensor_rank,:,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.mlp.dense_h_to_4h.weight'] = splited_mlp_h_to_4h_weight.clone()

                # MLP Dense h-to-4h Bias
                mlp_h_to_4h_bias = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.mlp.dense_h_to_4h.bias']
                mlp_h_to_4h_bias_shape = mlp_h_to_4h_bias.shape
                splited_mlp_h_to_4h_bias = mlp_h_to_4h_bias.view(tensor_num, -1)[tensor_rank,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.mlp.dense_h_to_4h.bias'] = splited_mlp_h_to_4h_bias.clone()

                # MLP Dense 4h-to-h Weight
                mlp_4h_to_h_weight = whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.mlp.dense_4h_to_h.weight']
                mlp_4h_to_h_weight_shape = mlp_4h_to_h_weight.shape
                splited_mlp_4h_to_h_weight = mlp_4h_to_h_weight.view(mlp_4h_to_h_weight_shape[0], tensor_num, -1,)[:,tensor_rank,:]
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.mlp.dense_4h_to_h.weight'] = splited_mlp_4h_to_h_weight.clone()

                # MLP Dense 4h-to-h Bias
                splited_state_dict[f'language_model.transformer.layers.{layer_index}.mlp.dense_4h_to_h.bias'] = \
                    whole_state_dict[f'language_model.transformer.layers.{layer_index + start_layer_index}.self_attention.dense.bias']



            if is_first_stage:
                splited_state_dict['language_model.proj.weight'] = \
                    whole_state_dict['language_model.proj.weight']
                splited_state_dict['language_model.proj.bias'] = \
                    whole_state_dict['language_model.proj.bias']
                splited_state_dict['language_model.proj_bulk.weight'] = \
                    whole_state_dict['language_model.proj_bulk.weight']
                splited_state_dict['language_model.proj_bulk.bias'] = \
                    whole_state_dict['language_model.proj_bulk.bias']
                
                splited_state_dict['language_model.down_blk.conv.weight'] = \
                    whole_state_dict['language_model.down_blk.conv.weight']
                splited_state_dict['language_model.down_blk.conv.bias'] = \
                    whole_state_dict['language_model.down_blk.conv.bias']
                
                for i in [0,1,3,4]:
                    splited_state_dict[f'language_model.down_blk.b.{i}.weight'] = \
                        whole_state_dict[f'language_model.down_blk.b.{i}.weight']
                    splited_state_dict[f'language_model.down_blk.b.{i}.bias'] = \
                        whole_state_dict[f'language_model.down_blk.b.{i}.bias']
                
                splited_state_dict['language_model.down_blk2.conv.weight'] = \
                    whole_state_dict['language_model.down_blk2.conv.weight']
                splited_state_dict['language_model.down_blk2.conv.bias'] = \
                    whole_state_dict['language_model.down_blk2.conv.bias']
                
                for i in [0,1,3,4]:
                    splited_state_dict[f'language_model.down_blk2.b.{i}.weight'] = \
                        whole_state_dict[f'language_model.down_blk2.b.{i}.weight']
                    splited_state_dict[f'language_model.down_blk2.b.{i}.bias'] = \
                        whole_state_dict[f'language_model.down_blk2.b.{i}.bias']
            
            if is_last_stage:
                splited_state_dict['language_model.up_blk.conv.weight'] = \
                    whole_state_dict['language_model.up_blk.conv.weight']
                splited_state_dict['language_model.up_blk.conv.bias'] = \
                    whole_state_dict['language_model.up_blk.conv.bias']
                
                for i in [0,1,3,4]:
                    splited_state_dict[f'language_model.up_blk.b.{i}.weight'] = \
                        whole_state_dict[f'language_model.up_blk.b.{i}.weight']
                    splited_state_dict[f'language_model.up_blk.b.{i}.bias'] = \
                        whole_state_dict[f'language_model.up_blk.b.{i}.bias']
                
                splited_state_dict['language_model.fc.weight'] = \
                    whole_state_dict['language_model.fc.weight']
                splited_state_dict['language_model.fc.bias'] = \
                    whole_state_dict['language_model.fc.bias']

            print(pipeline_rank, splited_state_dict.keys())
            return splited_state_dict

        directory_name = 'release'

        if pipeline_split is not None:
            pipeline_num = len(pipeline_split)

        pipeline_parallel = True if pipeline_num > 1 else False
        tensor_parallel = True if tensor_num > 1 else False

        for tensor_rank in range(tensor_num):
            for pipeline_rank in range(pipeline_num):
                splited_state_dict = split_for_parallel(whole_state_dict, tensor_rank, pipeline_rank, tensor_num, pipeline_num, pipeline_split)

                state_dict = {
                    'args': None,
                    'checkpoint_version': 3.0,
                    'iteration': 0,
                    'model': splited_state_dict,
                    'rng_state': None
                }

                if not pipeline_parallel:
                    common_path = os.path.join(new_model_path, directory_name, f'mp_rank_{tensor_rank:02d}')
                else:
                    common_path = os.path.join(new_model_path, directory_name,
                                               f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
                
                os.makedirs(common_path, exist_ok=True)

                save_path = os.path.join(common_path, 'model_optim_rng.pt')
                print(f'saving {save_path}')
                torch.save(state_dict, save_path)

    os.makedirs(new_model_path, exist_ok=True)

    orig_state_dict = read_state_dict(orig_model_path)

    to_megatron_state_dict = convert_to_megatron_state_dict(orig_state_dict, tensor_num)
    # print(to_megatron_state_dict.keys())
    # print(to_megatron_state_dict['language_model.transformer.layers.0.input_layernorm.weight'])
    split_and_save_for_parallel(new_model_path, to_megatron_state_dict, tensor_num, pipeline_num, layers)

    tracker_filename = os.path.join(new_model_path, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'w') as f:
        f.write('release')
    f.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="/test2/cuiyz/models/Wenhai1.0_finetune_S5-40/run_Sx5_surface_bs16/ckpt/epoch19.pth",
        help="Location of origin Wenhai weights, which should be a .pth file",
    )
    parser.add_argument(
        "--layers", default="10", type=int,
        choices=["10", "12"],
    )
    parser.add_argument(
        "--output-dir", default="./checkpoint/wenhai_ftep19_pp262",
        help="Location to write megatron version model and tokenizer",
    )
    parser.add_argument(
        "--pipeline-num", default="4", type=int
    )
    parser.add_argument(
        "--tensor-num", default="1", type=int
    )
    parser.add_argument('--inbalance-pipeline-stage', type=int, nargs='+')
    
    args = parser.parse_args()
    print(f"{args.inbalance_pipeline_stage}")
    convert_model(
        new_model_path=args.output_dir,
        orig_model_path=args.input_dir,
        layers=args.layers,
        tensor_num=args.tensor_num,
        pipeline_num=args.pipeline_num,
        pipeline_split=args.inbalance_pipeline_stage
    )


if __name__ == "__main__":
    main()