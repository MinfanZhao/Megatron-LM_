# generate torch pth for chinese llama

# python ./tools/llama/convert_llama_pth_from_hf.py \
#     --base_model /staff/zzq/code/LLM-Deploy/llama-chinese/merged/chinese-llama-alpaca-plus-33B \
#     --output_type pth \
#     --output_dir ./checkpoints/torch-pth/chinese-llama-alpaca-plus/30B

# python ./tools/llama/checkpoint_convert_from_llama_pth.py \
#     --input_dir ./checkpoints/torch-release/chinese-llama-alpaca-plus-7B-p1t8-pad1 \
#     --output_dir ./checkpoints/megatron-pretrain/chinese-llama-alpaca-plus-7B-p1t4-pad1-from-pad1 \
#     --model_size 7B \
#     --pipeline_size 1 \
#     --tensor_size 4 \
#     --make-vocab-size-divisible-by 128

# python ./tools/llama/checkpoint_convert_from_llama_pth.py \
#     --input_dir ./checkpoints/torch-release/chinese-llama-alpaca-plus-13B-p2t4-sft-sq2048-mbs2-gbs128-pad128 \
#     --output_dir ./checkpoints/megatron-pretrain/chinese-llama-alpaca-plus-13B-p2t4-sft-sq2048-mbs2-gbs128-pad128-p1t8 \
#     --model_size 13B \
#     --pipeline_size 1 \
#     --tensor_size 8 \
#     --make-vocab-size-divisible-by 128

python ./tools/llama/checkpoint_convert_from_llama_pth.py \
    --input_dir /acsa-med/megatron-llama/torch-pretrain/chinese-llama-alpaca-plus \
    --output_dir /acsa-med/megatron-llama/megatron-pretrain/chinese-llama-alpaca-plus-33B-p10t4 \
    --model_size 30B \
    --pipeline_size 10 \
    --tensor_size 4 \
    --make-vocab-size-divisible-by 130