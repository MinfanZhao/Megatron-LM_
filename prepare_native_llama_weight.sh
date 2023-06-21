python ./tools/llama/checkpoint_convert_from_llama_pth.py --output_dir ./checkpoints/llama-7B-p1t4 --pipeline_size 1 --tensor_size 4
python ./tools/llama/checkpoint_convert_from_llama_pth.py --output_dir ./checkpoints/llama-7B-p2t2 --pipeline_size 2 --tensor_size 2
python ./tools/llama/checkpoint_convert_from_llama_pth.py --output_dir ./checkpoints/llama-7B-p4t1 --pipeline_size 4 --tensor_size 1