MODEL_NAME=chinese-llama-alpaca-plus-13B-p2t4-sft-sq2048-mbs2-gbs128-pad128
# llama-7B-sw-0p8e-fp32
INPUT_MODEL_DIR=./checkpoints/megatron-train/$MODEL_NAME
CLEAN_MODEL_DIR=./checkpoints/megatron-clean/$MODEL_NAME
OUTPUT_MODEL_DIR=./checkpoints/torch-release/$MODEL_NAME
# OUTPUT_MODEL_DIR=/staff/zzq/checkpoints
HF_MODEL_DIR=./checkpoints/huggingface-release/$MODEL_NAME
MEGATRON_PATH=./
PIPELINE_SIZE=2
TENSOR_SIZE=4
MODEL_SIZE=13B

# python ./tools/llama/clean_checkpoint.py \
#     --megatron_path $MEGATRON_PATH \
#     --input_dir  $INPUT_MODEL_DIR \
#     --output_dir $CLEAN_MODEL_DIR \
#     --pipeline_size $PIPELINE_SIZE \
#     --tensor_size $TENSOR_SIZE \
#     --to_release

echo " > convert to llama pth ... "

python ./tools/llama/checkpoint_convert_to_llama_pth.py \
    --megatron_path $MEGATRON_PATH \
    --input_dir $CLEAN_MODEL_DIR \
    --output_dir $OUTPUT_MODEL_DIR \
    --model_size $MODEL_SIZE \
    --pipeline_size $PIPELINE_SIZE \
    --tensor_size $TENSOR_SIZE   

cp ./checkpoints/torch-pretrain/chinese-llama-alpaca-plus/$MODEL_SIZE/params.json $OUTPUT_MODEL_DIR/$MODEL_SIZE/
cp ./checkpoints/torch-pretrain/chinese-llama-alpaca-plus/$MODEL_SIZE/tokenizer.model $OUTPUT_MODEL_DIR/
cp ./checkpoints/torch-pretrain/chinese-llama-alpaca-plus/$MODEL_SIZE/special_tokens_map.json $OUTPUT_MODEL_DIR/

echo " > convert to llama huggingface ... "

python ./tools/llama/convert_llama_pth_to_hf.py \
    --input_dir $OUTPUT_MODEL_DIR \
    --output_dir $HF_MODEL_DIR \
    --model_size 13B
