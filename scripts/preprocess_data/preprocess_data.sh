python tools/preprocess_data.py \
    --input /acsa-med/dataset/sunway/chinese-pretrain-raw/code-clippy-all/test_code.json \
    --tokenizer-type LLaMASentencePieceTokenizer \
    --tokenizer-model checkpoints/tokenizer/chinese-llama/tokenizer.model \
    --workers 40 \
    --chunk-size 40 \
    --log-interval 1000 \
    --append-eod \
    --output-prefix /acsa-med/dataset/sunway/chinese-pretrain/code-clippy-all/test_code

