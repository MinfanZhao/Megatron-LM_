# python tools/preprocess_data.py \
#     --input /staff/zzq/dataset/nlp/sw_data/Marxism-cn \
#     --tokenizer-type LLaMASentencePieceTokenizer \
#     --tokenizer-model checkpoints/tokenizer/chinese-llama/tokenizer.model \
#     --workers 80 \
#     --chunk-size 25 \
#     --log-interval 1000 \
#     --append-eod \
#     --output-prefix /acsa-med/dataset/sunway/chinese-pretrain/Marxism-cn/0921

# python tools/preprocess_data.py \
#     --input /staff/zzq/dataset/nlp/sw_data/Wikipedia/wiki-cn \
#     --tokenizer-type LLaMASentencePieceTokenizer \
#     --tokenizer-model checkpoints/tokenizer/chinese-llama/tokenizer.model \
#     --workers 80 \
#     --chunk-size 25 \
#     --log-interval 1000 \
#     --append-eod \
#     --output-prefix /acsa-med/dataset/sunway/chinese-pretrain/wikipedia-cn/0921

python tools/preprocess_data.py \
    --input /staff/zzq/dataset/nlp/sw_data/code-clippy-all/test_code.json \
    --tokenizer-type LLaMASentencePieceTokenizer \
    --tokenizer-model checkpoints/tokenizer/chinese-llama/tokenizer.model \
    --workers 80 \
    --chunk-size 25 \
    --log-interval 1000 \
    --append-eod \
    --output-prefix /acsa-med/dataset/sunway/chinese-pretrain/test_code