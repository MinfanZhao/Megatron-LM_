python tools/preprocess_data.py \
    --input /staff/zzq/dataset/nlp/sw_data/WuDaoCorpus2.0_base_200G \
    --tokenizer-type LLaMASentencePieceTokenizer \
    --tokenizer-model checkpoints/tokenizer/chinese-llama/tokenizer.model \
    --workers 80 \
    --chunk-size 25 \
    --log-interval 1000 \
    --append-eod \
    --output-prefix /acsa-med/dataset/sunway/chinese-pretrain/wudao/0921
