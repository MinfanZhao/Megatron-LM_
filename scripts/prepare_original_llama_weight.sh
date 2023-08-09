python ./tools/llama/checkpoint_convert_from_llama_pth.py \
    --input_dir ./checkpoints/checkpoint-llama \
    --model_size 13B \
    --output_dir ./checkpoints/llama-13B-p2t2-pad128 \
    --pipeline_size 2 \
    --tensor_size 2 \
    --make-vocab-size-divisible-by 128