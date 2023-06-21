python tools/preprocess_data.py \
    --input /staff/zzq/dataset/nlp/llama/arxiv_sample.jsonl \
    --tokenizer-type LLaMASentencePieceTokenizer \
    --tokenizer-model checkpoints/chinese-llama/7B/tokenizer.model \
    --workers 80 \
    --chunk-size 1 \
    --output-prefix data/llama-dataset/arxiv_sentence_chinese_tokenizer \
    --split-sentences \
    --append-bos \
    --append-eos \
    # --append-eod \
    # llama_tokenizer.model \