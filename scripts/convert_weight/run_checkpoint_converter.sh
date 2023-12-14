python tools/checkpoint_converter.py \
    --megatron-path '.' \
    --input-dir '/ustc1/checkpoint/ipt-13B/ipt-13B-p1t1' \
    --output-dir '/ustc1/checkpoint/ipt-13B/ipt-13B-p2t4' \
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --gate-gelu \
    --load-pipeline-model-parallel-size 1 \
    --load-tensor-model-parallel-size 1 \
    --save-pipeline-model-parallel-size 2 \
    --save-tensor-model-parallel-size 4 \
    --vocab-size 60000 \
    --make-vocab-size-divisible-by 4 \
    