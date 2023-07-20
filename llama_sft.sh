#! /bin/bash

NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR=hades0.acsalab.com
# 10.1.13.63
MASTER_PORT=25934

NODE_RANK=0
pwd
# 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=./checkpoints/chinese-llama-7B-p2t4-pad128/
TASK_NAME=chinese-llama-7B-p2t4-sft-sq2048-mbs2-gbs128-pad128
SAVE_CHECKPOINT_PATH=./checkpoints/$TASK_NAME/
TENSORBOARD_PATH=./tensorboard/megatron/$TASK_NAME/$DATETIME

TOKENIZER_MODEL="./checkpoints/chinese-llama/7B/tokenizer.model"
TRAIN_DATA_PATH=/acsa-med/dataset/sunway/iflytek_digital/right_padding/train_data_for_SW_token_ids_2048.json
TEST_DATA_PATH=/acsa-med/dataset/sunway/iflytek_digital/right_padding/test_data_for_SW_token_ids_2048.json

# args for llama 7B
MODEL_ARGS="--num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048"

LLAMA_ARGS="--use-rmsnorm \
        --swiglu \
        --llama-swiglu \
        --rope-style llama \
        --no-query-key-layer-scaling \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --disable-bias-linear \
        --untie-embeddings-and-output-weights"

TRAIN_ARGS="--lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters 8000 \
        --lr-warmup-iters 400 \
        --lr 2e-05 \
        --min-lr 2e-06 \
        --micro-batch-size 2 \
        --global-batch-size 128"

EVAL_ARGS="--eval-interval 1000 \
        --eval-iters 20"

OUTPUT_ARGS="--log-interval 1 \
        --save-interval 2000"

DATASET_ARGS="--train-data-path $TEST_DATA_PATH \
        --test-data-path $TEST_DATA_PATH \
        --data-length 2049 \
        --split 100,0,0 \
        --num-workers 8 \
        --padding-direction right "


CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./tasks/main.py \
        --task LLAMA_SFT \
        --epochs 2 \
        --pretrained-checkpoint $LOAD_CHECKPOINT_PATH \
        --tokenizer-type LLaMASentencePieceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 2 \
        $MODEL_ARGS \
        $LLAMA_ARGS \
        $TRAIN_ARGS \
        $EVAL_ARGS \
        $OUTPUT_ARGS \
        $DATASET_ARGS \
        --init-method-std 0.002 \
        --make-vocab-size-divisible-by 128 \
        --DDP-impl local \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --fp16 \
        --use-flash-attn \
        --initial-loss-scale 32768.0 \
        --no-load-optim \
        --no-load-rng \
        --finetune
        