python tools/preprocess_data.py \
    --input /home/nfs/zzq/data/llama-data/arxiv_sample.jsonl \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model llama_tokenizer.model \
    --workers 80 \
    --chunk-size 1 \
    --output-prefix data/llama-dataset/arxiv_sample