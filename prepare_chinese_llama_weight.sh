python ./tools/llama/convert_llama_pth_from_hf.py \
    --base_model ./checkpoints/ymcui-chinese-llama-alpaca-plus-7B \
    --output_type pth \
    --output_dir ./checkpoints/chinese-llama/7B

python ./tools/llama/checkpoint_convert_from_llama_pth.py \
    --input_dir ./checkpoints/chinese-llama \
    --output_dir ./checkpoints/chinese-llama-7B-p2t2 \
    --pipeline_size 2 \
    --tensor_size 2