#!/bin/bash
module load gcc/7.3.0-os7.2
module unload cuda/8.0
module load cuda/11.7.1-cudnn-v8.5.0
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home4/intern/mfzhao4/testdata/
CHECKPOINT_PATH=/home4/intern/mfzhao4/testckpt/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_mae.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --encoder-num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --decoder-num-layers 8 \
       --decoder-embed-dim 512 \
       --decoder-ffn-size 2048 \
       --kv-channels-decoder 32 \
       --decoder-num-attention-heads 16 \
       --vision-pretraining \
       --vision-pretraining-type mae \
       --vision-backbone-type mae \
       --num-classes 3 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --train-iters 1000 \
       --lr-decay-iters 1000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 10 \
       --save-interval 10 \
       --fp16  \
       --max-position-embeddings 512 \
       --seq-length 196 \
       --pipeline-model-parallel-split-rank 1 \
