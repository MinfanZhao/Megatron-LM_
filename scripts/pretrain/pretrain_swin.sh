#! /bin/bash

# Runs the vit model

GPUS_PER_NODE=2
DEVICES=0,1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`


DATA_PATH="/acsa-med/radiology/ImageNet-1K/ILSVRC2012/train \
/acsa-med/radiology/ImageNet-1K/ILSVRC2012/val"
#$(cat imagenet_2012_datapath.txt)
CHECKPOINT_PATH=./checkpoint/swin_transformer/imagenet_1k/test/
TENSORBOARD_PATH=./tensorboard/swin_transformer/imagenet_1k/test/$DATETIME

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                     --nnodes $NNODES \
                     --node_rank $NODE_RANK \
                     --master_addr $MASTER_ADDR \
                     --master_port $MASTER_PORT"

DATA_ARGS="--data-path $DATA_PATH \
              --num-classes 1000 \
              --img-size 224\
              --num-workers 32"


REQUIRED_ARGS="--num-attention-heads 1 \
              --max-position-embeddings 0 \
              --seq-length 0"

SWIN_ARGS="--hidden-size 768 \
           --patch-size 4 \
           --depths 10 \
           --window-size 7 \
           --num-heads 12 \
           --patch-norm True \
           --in-chans 3 \
           --drop-rate 0 \
           --attn-drop-rate 0 \
           --drop-path-rate 0.2 \
           --ape False \
           --constant-drop-path-rate True"



TRAINING_ARGS="--micro-batch-size 2 \
              --global-batch-size 4 \
              --train-iters 1000 \
              --log-interval 10
              --eval-interval 100 \
              --optimizer adam \
              --dataloader-type cyclic\
              --init-method-xavier-uniform"

LOGGING_ARGS="--tensorboard-log-interval 1 \
              --log-num-zeros-in-grad \
              --log-params-norm"

REGULARIZATION_ARGS="--clip-grad 1.0 \
                     --weight-decay 0.05 \
                     --adam-beta1 0.9 \
                     --adam-beta2 0.999"

LEARNING_RATE_ARGS=" --lr 4.0e-3 \
                     --lr-decay-style cosine \
                     --lr-decay-iters 31250 \
                     --lr-warmup-fraction 0.05 \
                     --min-lr 1.0e-5"

MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 1 \
                     --pipeline-model-parallel-size 2"

MIXED_PRECISION_ARGS="--fp16"



CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=$DEVICES \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_swin.py \
       $SWIN_ARGS \
       $MODEL_PARALLEL_ARGS \
       $REQUIRED_ARGS \
       $DATA_ARGS \
       $TRAINING_ARGS \
       $REGULARIZATION_ARGS \
       $LEARNING_RATE_ARGS \
       $MODEL_PARALLEL_ARGS \
       $MIXED_PRECISION_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --save-interval 312 \
       --eval-iters 10 \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 10 \
       --distributed-backend nccl \
       --init-method-xavier-uniform 