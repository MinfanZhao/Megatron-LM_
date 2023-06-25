python ./tools/llama/checkpoint_convert_from_llama_pth.py \
    --input_dir ./checkpoints/checkpoint-llama \
    --model_size 7B \
    --output_dir ./checkpoints/llama-7B-p2t2 \
    --pipeline_size 2 \
    --tensor_size 2