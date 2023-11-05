#! /bin/bash

NNODES=1
GPUS_PER_NODE=1
NODE_RANK=$1
MASTER_ADDR=hades0.acsalab.com
MASTER_PORT=25936
pwd

# Distributed environment

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"


TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
VOCAB_PAD_SIZE=128

MODEL_NAME=llama-toy
TASK_FLAG=test-fp16
TASK_NAME=$MODEL_NAME-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-sq2048-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-pad${VOCAB_PAD_SIZE}-${TASK_FLAG}


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=./checkpoints/megatron-train/$TASK_NAME-checkpoint/
#extend-meta-llama-7B-p1t2-pad128
# /acsa-med/megatron-llama/megatron-train/$TASK_NAME/
# 
#-sq2048-mbs1-gbs128-pad128-wanjuan+pile-scratch #extend-meta-llama-7B-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-pad${VOCAB_PAD_SIZE}
SAVE_CHECKPOINT_PATH=./checkpoints/megatron-train/$TASK_NAME/
# ./checkpoints/megatron-train/$TASK_NAME/
TENSORBOARD_PATH=./tensorboard/llama-toy-test/$TASK_NAME/$DATETIME

# args for llama 7B
MODEL_ARGS="--num-layers 4 \
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
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-samples 38400 \
        --lr-warmup-samples 2048 \
        --train-samples 51200 \
        --lr 1.6e-07 \
        --min-lr 1.6e-08 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --init-method-std 0.002"


EVAL_ARGS="--eval-interval 1000 \
        --eval-iters 20"

OUTPUT_ARGS="--log-interval 1 \
        --save-interval 1"

LOG_ARGS="--tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1"



CHECKPOINT_ARGS="--load $LOAD_CHECKPOINT_PATH \
        --save $SAVE_CHECKPOINT_PATH"


TOKENIZER_MODEL=./checkpoints/tokenizer/chinese-llama/tokenizer.model
DATA_PATH="0.00162911 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider13_text_document \
0.00225660 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider11_text_document"
# 0.00225531 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider5_text_document \
# 0.00225526 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider7_text_document \
# 0.00226065 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider2_text_document \
# 0.00182835 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider9_text_document \
# 0.00190775 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider12_text_document \
# 0.00191197 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider8_text_document \
# 0.00224886 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider1_text_document \
# 0.00225682 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider4_text_document \
# 0.00220159 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider10_text_document \
# 0.00225159 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider6_text_document \
# 0.00223032 /acsa-med/dataset/sunway/chinese-pretrain/baidu-baike/0921_spider3_text_document"


DATASET_ARGS="--data-path $DATA_PATH \
        --split 95,5,0 \
        --num-workers 0 \
        --data-impl mmap \
        --reset-attention-mask \
        --reset-position-ids \
        --eod-mask-loss"


CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_llama.py \
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
        --initial-loss-scale 32768.0 \
        --fp16 \
        # --use-flash-attn \
        # --sequence-parallel \
        # --finetune
        # \
        # 
        
        
        