#! /bin/bash

NNODES=4
GPUS_PER_NODE=2
MASTER_ADDR=icarus0.acsalab.com
# 10.1.13.63
MASTER_PORT=25934

NODE_RANK=$1

# 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=./checkpoints/test_llama/
SAVE_CHECKPOINT_PATH=./checkpoints/test_llama/
TENSORBOARD_PATH=./tensorboard/test_llama/$DATETIME

# args for llama 7B
MODEL_ARGS="--num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 2048"

LLAMA_ARGS="--use-rmsnorm \
        --swiglu \
        --llama-swiglu \
        --rope-style llama"

TRAIN_ARGS="--max-position-embeddings 2048 \
        --lr-decay-samples 258560 \
        --lr-warmup-samples 248800 \
        --lr 1.0e-04 \
        --min-lr 1.0e-05 \
        --lr-decay-style cosine \
        --micro-batch-size 1 \
        --global-batch-size 32 \
        --train-samples 488280"

OUTPUT_ARGS="--log-interval 1 \
        --eval-iters -1 \
        --save-interval 300"

TOKENIZER_MODEL="./llama_tokenizer.model"
# DATA_PATH="/staff/zzq/dataset/nlp/Yuan_Processed/001.txt_document_context"
DATA_PATH="/home/nfs/zzq/data/asc/001.txt_document_context"

CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_llama.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        $MODEL_ARGS \
        $LLAMA_ARGS \
        $TRAIN_ARGS \
        $OUTPUT_ARGS \
        --data-path ${DATA_PATH} \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --DDP-impl local \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --use-distributed-optimizer \
        --use-flash-attn \
        # --no-gradient-accumulation-fusion \
        # --sequence-parallel \
        # --checkpoint-num-layers 1 \
        # --encoder-attn-mask-type padding \
        # --checkpoint-activations \