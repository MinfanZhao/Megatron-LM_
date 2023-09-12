python tools/llama/clean_checkpoint.py \
    --megatron_path ./ \
    --input_dir checkpoints/chinese-llama-7B-p4t1-sq1024-lrdecay30000-fixloss \
    --output_dir checkpoints/chinese-llama-7B-p4t1-sq1024-clean \
    --pipeline_size 4 \
    --tensor_size 1 \
    --to_release