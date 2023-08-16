import sys
import os
import argparse
import torch
    
def clean_state_dict(megatron_model_load_base, megatron_model_save_base, 
                     pipeline_parallel_size, tensor_parallel_size, save_optim, save_rng, to_release):
    tracker_filename = os.path.join(megatron_model_load_base, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'r') as f:
        iteration = f.read().strip()
        if iteration == 'release':
            megatron_model_load_base = os.path.join(
                megatron_model_load_base, 'release')
        else:
            iteration = int(iteration)
            megatron_model_load_base = os.path.join(
                megatron_model_load_base, 'iter_{:07d}'.format(iteration))
    
    
    os.makedirs(megatron_model_save_base, exist_ok=True)
    save_tracker_filename = os.path.join(megatron_model_save_base, 'latest_checkpointed_iteration.txt')
    
    if to_release:
        with open(save_tracker_filename, 'w') as f:
            f.write('release')
        megatron_model_save_base = os.path.join(
            megatron_model_save_base, 'release')
    else:
        with open(save_tracker_filename, 'w') as f:
            f.write(str(iteration))
        megatron_model_save_base = os.path.join(
            megatron_model_save_base, 'iter_{:07d}'.format(iteration))
    
    
    pipeline_parallel = pipeline_parallel_size > 1
    for pipeline_rank in range(pipeline_parallel_size):
        for tensor_rank in range(tensor_parallel_size):
            if not pipeline_parallel:
                common_path = f'mp_rank_{tensor_rank:02d}'          
            else:
                common_path = f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}'
            load_checkpoint_path = os.path.join(
                megatron_model_load_base, common_path, 'model_optim_rng.pt')
            print(f"loading {load_checkpoint_path}")
            state_dict = torch.load(load_checkpoint_path, map_location="cpu")
            if not save_optim:
                state_dict['optimizer'] = None
                state_dict['opt_param_scheduler'] = None
            if not save_rng:
                state_dict['rng_state'] = None
            
            save_base = os.path.join(megatron_model_save_base, common_path)
            os.makedirs(save_base, exist_ok=True)
            
            save_checkpoint_path = os.path.join(
                save_base, "model_optim_rng.pt")
            print(f"saving {save_checkpoint_path}")
            torch.save(state_dict, save_checkpoint_path)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--megatron_path', type=str, default=None,
                       help='Base directory of Megatron repository')
    parser.add_argument(
        "--input_dir", default="./checkpoints/checkpoint-llama",
        help="Location of Megatron weights",
    )
    parser.add_argument(
        "--output_dir", default="./checkpoints/llama-7B-p1t4/",
        help="Location to write clean version model of Megatron",
    )
    parser.add_argument(
        "--pipeline_size", default="1", type=int
    )
    parser.add_argument(
        "--tensor_size", default="4", type=int
    )
    parser.add_argument(
        "--save_optim", default=False, action="store_true"
    )
    parser.add_argument(
        "--save_rng", default=False, action="store_true"
    )
    parser.add_argument(
        "--to_release", default=False, action="store_true"
    )
    args = parser.parse_args()
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
        
    clean_state_dict(
        megatron_model_load_base=args.input_dir,
        megatron_model_save_base=args.output_dir,
        pipeline_parallel_size=args.pipeline_size,
        tensor_parallel_size=args.tensor_size,
        save_optim=args.save_optim,
        save_rng=args.save_rng,
        to_release=args.to_release
        )
    
if __name__ == "__main__":
    main()