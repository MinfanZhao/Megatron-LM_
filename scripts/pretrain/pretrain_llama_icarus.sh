#! /bin/bash

NNODES=4
GPUS_PER_NODE=2
NODE_RANK=$1
MASTER_ADDR=icarus0.acsalab.com
MASTER_PORT=25935
pwd

# Distributed environment

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

echo "distributed args: $DISTRIBUTED_ARGS"


TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=4

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
VOCAB_PAD_SIZE=128

MODEL_NAME=extend-meta-llama-7B
TASK_FLAG=wanjuan+pile-scratch-continue-seq-parallel
TASK_NAME=$MODEL_NAME-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-sq2048-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-pad${VOCAB_PAD_SIZE}-${TASK_FLAG}


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=/acsa-med/megatron-llama/megatron-train/extend-meta-llama-7B-p4t2-sq2048-mbs1-gbs128-pad128-wanjuan+pile-scratch #extend-meta-llama-7B-p${PIPELINE_MODEL_PARALLEL_SIZE}t${TENSOR_MODEL_PARALLEL_SIZE}-pad${VOCAB_PAD_SIZE}
SAVE_CHECKPOINT_PATH=/acsa-med/megatron-llama/megatron-train/$TASK_NAME/
TENSORBOARD_PATH=./tensorboard/extend-meta-llama-7B/$TASK_NAME/$DATETIME


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
        --lr-decay-samples 384000 \
        --lr-warmup-samples 20480 \
        --train-samples 512000 \
        --lr 2e-05 \
        --min-lr 2e-06 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --init-method-std 0.002"


EVAL_ARGS="--eval-interval 500 \
        --eval-iters 20"

OUTPUT_ARGS="--log-interval 5 \
        --save-interval 500"

LOG_ARGS="--tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1"

# --log-timers-to-tensorboard \
        # --timing-log-level 2\
        # --log-num-zeros-in-grad \
        # --log-params-norm \

CHECKPOINT_ARGS="--load $LOAD_CHECKPOINT_PATH \
        --save $SAVE_CHECKPOINT_PATH"
        #  \
        # --no-load-optim \
        # --no-load-rng"


TOKENIZER_MODEL=./checkpoints/tokenizer/chinese-llama/tokenizer.model
DATA_PATH="0.00709960 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/ChinaNews-cn/0921_part-008323-4fcc5597-xac_text_document \
0.00705630 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/ChinaNews-cn/0921_part-008323-4fcc5597-xae_text_document \
0.00709674 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/ChinaNews-cn/0921_part-008323-4fcc5597-xab_text_document \
0.00711250 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/ChinaNews-cn/0921_part-008323-4fcc5597-xad_text_document \
0.00714064 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/ChinaNews-cn/0921_part-008323-4fcc5597-xaa_text_document \
0.03663060 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Law-cn/0921_part-001744-4fcc5597_text_document \
0.03661836 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Law-cn/0921_part-009942-4fcc5597_text_document \
0.03657915 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Law-cn/0921_part-007119-4fcc5597_text_document \
0.03665461 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Law-cn/0921_part-001398-4fcc5597_text_document \
0.01140872 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Patent-cn/_part-008323-4fcc5597_text_document \
0.01139829 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Patent-cn/_part-003260-4fcc5597_text_document \
0.00294262 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/TextBook-cn/0921_part-009273-4fcc5597_text_document \
0.05012058 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-001199-4fcc5597_text_document \
0.05011552 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-000036-4fcc5597_text_document \
0.05009851 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-002169-4fcc5597_text_document \
0.05006101 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-000520-4fcc5597_text_document \
0.05009489 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-000467-4fcc5597_text_document \
0.05010046 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-000895-4fcc5597_text_document \
0.05004889 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-001772-4fcc5597_text_document \
0.05005897 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-001819-4fcc5597_text_document \
0.05014245 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-001835-4fcc5597_text_document \
0.05009973 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-cn/0921_part-000122-4fcc5597_text_document \
0.00053284 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/Wiki-cn/0921_part-006656-4fcc5597_text_document \
0.02189047 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000369-4fcc5597_text_document \
0.02189279 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000020-4fcc5597_text_document \
0.02191330 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000419-4fcc5597_text_document \
0.02191031 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000442-4fcc5597_text_document \
0.02193115 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000341-4fcc5597_text_document \
0.02189066 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000065-4fcc5597_text_document \
0.02192909 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000363-4fcc5597_text_document \
0.02190570 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000376-4fcc5597_text_document \
0.02190434 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000286-4fcc5597_text_document \
0.02191432 /acsa-med/dataset/sunway/chinese-pretrain/wanjuan/WebText-en/0921_part-000346-4fcc5597_text_document \
0.00466279 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00009-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00465827 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00006-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00464083 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00005-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00468625 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00007-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00463994 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00004-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00465974 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00003-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00465339 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00008-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00467706 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00002-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00469318 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00001-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00468186 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-cc/0921_part-00000-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00255740 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00397-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00288883 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00358-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00229680 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00445-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00253814 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00401-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00245691 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00419-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00247186 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00412-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00254395 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00400-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00234585 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00439-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00239399 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00429-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document \
0.00255887 /acsa-med/dataset/sunway/chinese-pretrain/pile/pile-owt/0921_part-00399-1ada68f1-85b8-473a-b6af-71a367a5ccbf_text_document"

# DATA_PATH=


DATASET_ARGS="--data-path $DATA_PATH \
        --split 95,4,1 \
        --num-workers 16"


CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1 \
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
        --initial-loss-scale 4096.0 \
        --use-flash-attn \
        --fp16 \
        --finetune \
        --sequence-parallel
        
        