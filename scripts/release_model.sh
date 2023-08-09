MODEL_NAME=chinese-llama-7B-p4t1-sft-sq2048-mbs2-gbs128-0716
INPUT_MODEL_DIR=./checkpoints/$MODEL_NAME
CLEAN_MODEL_DIR=./checkpoints/clean/$MODEL_NAME
OUTPUT_MODEL_DIR=./checkpoints/pth/$MODEL_NAME
HF_MODEL_DIR=./checkpoints/hf/$MODEL_NAME
MEGATRON_PATH=./
PIPELINE_SIZE=4
TENSOR_SIZE=1
MODEL_SIZE=7B

python ./tools/llama/clean_checkpoint.py \
    --megatron_path $MEGATRON_PATH \
    --input_dir  $INPUT_MODEL_DIR \
    --output_dir $CLEAN_MODEL_DIR \
    --pipeline_size $PIPELINE_SIZE \
    --tensor_size $TENSOR_SIZE \
    --to_release

echo " > convert to llama pth ... "

python ./tools/llama/checkpoint_convert_to_llama_pth.py \
    --megatron_path $MEGATRON_PATH \
    --input_dir $CLEAN_MODEL_DIR \
    --output_dir $OUTPUT_MODEL_DIR \
    --model_size $MODEL_SIZE \
    --pipeline_size $PIPELINE_SIZE \
    --tensor_size $TENSOR_SIZE   

cp ./checkpoints/checkpoint-llama/$MODEL_SIZE/params.json $OUTPUT_MODEL_DIR/$MODEL_SIZE/
cp ./checkpoints/ymcui-chinese-llama-alpaca-plus-7B/tokenizer.model $OUTPUT_MODEL_DIR
cp ./checkpoints/ymcui-chinese-llama-alpaca-plus-7B/special_tokens_map.json $OUTPUT_MODEL_DIR

echo " > convert to llama huggingface ... "

python ./tools/llama/convert_llama_pth_to_hf.py \
    --input_dir $OUTPUT_MODEL_DIR \
    --output_dir $HF_MODEL_DIR \
    --model_size 7B