#! /bin/bash

NNODES=1
GPUS_PER_NODE=4
MASTER_ADDR=hades0.acsalab.com
# 10.1.13.63
MASTER_PORT=25935

NODE_RANK=$1

# 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`


TASK_NAME=chinese-llama-7B-p2t2

LOAD_CHECKPOINT_PATH=./checkpoints/$TASK_NAME/
SAVE_CHECKPOINT_PATH=./checkpoints/$TASK_NAME-megatron/
TENSORBOARD_PATH=./tensorboard/$TASK_NAME/$DATETIME

# args for llama-7B 
MODEL_ARGS="--num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048"

LLAMA_ARGS="--use-rmsnorm \
        --swiglu \
        --llama-swiglu \
        --rope-style llama \
        --no-query-key-layer-scaling \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --disable-bias-linear \
        --untie-embeddings-and-output-weights"

TRAIN_ARGS="--max-position-embeddings 2048 \
        --lr-decay-samples 258560 \
        --lr-warmup-samples 248800 \
        --lr 1.0e-04 \
        --min-lr 1.0e-05 \
        --lr-decay-style cosine \
        --micro-batch-size 1 \
        --global-batch-size 1 \
        --train-samples 488280"

OUTPUT_ARGS="--log-interval 1 \
        --save-interval 300"

TOKENIZER_MODEL="./checkpoints/chinese-llama/7B/tokenizer.model"
# DATA_PATH="/staff/zzq/dataset/nlp/Yuan_Processed/001.txt_document_context"
DATA_PATH="data/llama-dataset/arxiv_sample_sentence_split_text_sentence"
#,1,2,3,4,5,6,7
CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_llama.py \
        --tokenizer-type LLaMASentencePieceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        $MODEL_ARGS \
        $LLAMA_ARGS \
        $TRAIN_ARGS \
        $OUTPUT_ARGS \
        --data-path ${DATA_PATH} \
        --split 95,4,1 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --DDP-impl local \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --no-load-optim \
        --no-load-rng \
        --finetune \
        --eval-interval 30 \
        --eval-iters 70 \
        --make-vocab-size-divisible-by 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --initial-loss-scale 32768.0 \
        --fp16 \
        --attention-softmax-in-fp32 
        # --use-flash-attn \
        # --no-masked-softmax-fusion \
        # --no-bias-gelu-fusion \
        # --no-bias-dropout-fusion \
        # --no-gradient-accumulation-fusion \
        