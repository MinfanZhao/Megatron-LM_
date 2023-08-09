# generate torch pth for chinese llama

# python ./tools/llama/convert_llama_pth_from_hf.py \
#     --base_model /staff/zzq/code/LLM-Deploy/llama-chinese/merged/chinese-llama-alpaca-plus-33B \
#     --output_type pth \
#     --output_dir ./checkpoints/torch-pth/chinese-llama-alpaca-plus/30B

python ./tools/llama/checkpoint_convert_from_llama_pth.py \
    --input_dir ./checkpoints/torch-pretrain/chinese-llama-alpaca-plus \
    --output_dir ./checkpoints/megatron-pretrain/chinese-llama-alpaca-plus-7B-p1t8-pad128 \
    --model_size 7B \
    --pipeline_size 1 \
    --tensor_size 8 \
    --make-vocab-size-divisible-by 128