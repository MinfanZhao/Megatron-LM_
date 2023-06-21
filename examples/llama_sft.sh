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
LOAD_CHECKPOINT_PATH=./checkpoints/test_llama_nopip/
SAVE_CHECKPOINT_PATH=./checkpoints/test_sft/
TENSORBOARD_PATH=./tensorboard/test_sft/$DATETIME

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
        --lr 1.0e-04 \
        --min-lr 1.0e-05 \
        --lr-decay-style cosine \
        --micro-batch-size 4 \
        --global-batch-size 4 \

OUTPUT_ARGS="--log-interval 1 \
        --eval-iters -1 \
        --save-interval 300"

TOKENIZER_MODEL="./llama_tokenizer.model"
# DATA_PATH="/staff/zzq/dataset/nlp/Yuan_Processed/001.txt_document_context"
DATA_PATH="/home/nfs/zzq/data/asc/001.txt_document_context"

CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./tasks/main.py \
        --task LLAMA_SFT \
        --finetune \
        --epochs 50 \
        --pretrained-checkpoint /staff/wangzhaohui/codes/Megatron-LM_/checkpoints/test_llama_nopip \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        $MODEL_ARGS \
        $LLAMA_ARGS \
        $TRAIN_ARGS \
        $OUTPUT_ARGS \
        --train-data-path /staff/zzq/dataset/nlp/for_sw/train_data_for_SW_token_ids.json \
        --test-data-path /staff/zzq/dataset/nlp/for_sw/test_data_for_SW_token_ids.json \
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
        # --use-distributed-optimizer \
        # --use-flash-attn \
        # --no-gradient-accumulation-fusion \
        # --sequence-parallel \
        # --checkpoint-num-layers 1 \
        # --encoder-attn-mask-type padding \
        # --checkpoint-activations \