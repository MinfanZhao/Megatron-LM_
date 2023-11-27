#! /bin/bash

NNODES=1
GPUS_PER_NODE=4
NODE_RANK=0
MASTER_ADDR=hades0.acsalab.com
MASTER_PORT=25935
pwd

# Distributed environment

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"


TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=1

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
VOCAB_PAD_SIZE=128

MODEL_NAME=chinese-llama-alpaca-plus-7B
TASK_NAME=$MODEL_NAME-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-sft-sq2048-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-pad${VOCAB_PAD_SIZE}


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=/acsa-med/megatron-llama/megatron-pretrain/chinese-llama-alpaca-plus-7B-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-pad${VOCAB_PAD_SIZE}
SAVE_CHECKPOINT_PATH=/acsa-med/megatron-llama/megatron-train/$TASK_NAME/
TENSORBOARD_PATH=./tensorboard/llama-test/$TASK_NAME/$DATETIME


# args for llama 7B
MODEL_ARGS="--num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048 \
        --max-position-embeddings 2048"

# args for llama 13B
# MODEL_ARGS="--num-layers 40 \
#         --hidden-size 5120 \
#         --num-attention-heads 40 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048"

# args for llama 33B
# MODEL_ARGS="--num-layers 60 \
#         --hidden-size 6656 \
#         --num-attention-heads 52 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048"

# args for llama 65B
# MODEL_ARGS="--num-layers 80 \
#         --hidden-size 8192 \
#         --num-attention-heads 64 \
#         --seq-length 2048 \
#         --max-position-embeddings 2048"

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
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --init-method-std 0.002"

EVAL_ARGS="--eval-interval 1000 \
        --eval-iters 20"

OUTPUT_ARGS="--log-interval 1 \
        --save-interval 2500"

LOG_ARGS="--tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1"

# --log-timers-to-tensorboard \
        # --timing-log-level 2\
        # --log-num-zeros-in-grad \
        # --log-params-norm \

CHECKPOINT_ARGS="--pretrained-checkpoint $LOAD_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --save $SAVE_CHECKPOINT_PATH \
        --no-load-optim \
        --no-load-rng"


TOKENIZER_MODEL=./checkpoints/tokenizer/chinese-llama/tokenizer.model
TRAIN_DATA_PATH=/acsa-med/dataset/sunway/iflytek_digital/right_padding/train_data_for_SW_token_ids_2048.json
TEST_DATA_PATH=/acsa-med/dataset/sunway/iflytek_digital/right_padding/test_data_for_SW_token_ids_2048.json


DATASET_ARGS="--train-data-path $TEST_DATA_PATH \
        --test-data-path $TEST_DATA_PATH \
        --data-length 2049 \
        --split 100,0,0 \
        --num-workers 16 \
        --padding-direction right "


CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./tasks/main.py \
        --task LLAMA_SFT \
        --epochs 2 \
        --tokenizer-type LLaMASentencePieceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
        --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
        --make-vocab-size-divisible-by $VOCAB_PAD_SIZE \
        $MODEL_ARGS \
        $LLAMA_ARGS \
        $TRAIN_ARGS \
        $EVAL_ARGS \
        $OUTPUT_ARGS \
        $DATASET_ARGS \
        $LOG_ARGS \
        $CHECKPOINT_ARGS \
        --DDP-impl local \
        --initial-loss-scale 4096.0 \
        --finetune \
        --use-flash-attn \
        --fp16
        
        