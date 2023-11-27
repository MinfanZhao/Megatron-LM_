import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.tokenizer import build_tokenizer

def check_tokenizer(tokenizer, check_string = "hello <end>"):
    print("eod id:", tokenizer.eod)
    print("bos id:", tokenizer.bos)
    print("eos id:", tokenizer.eos)
    # tokens = tokenizer.tokenize(check_string)
    # print(tokens)
    vocab = {}
    for token in range(49954):#tokens:
        vocab[token] = tokenizer.detokenize(token)
    with open("./tokenizer_vocab.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(vocab, ensure_ascii=False))
    print(vocab)
    # print(tokenizer.detokenize())
    # print(tokenizer.tokenize("hello <EOD>"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 
                                'GPTSentencePieceTokenizer', 'NullTokenizer',
                                'LLaMASentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-bos', action='store_true',
                       help='Append an <s> token to the begin of a sentence.')
    group.add_argument('--append-eos', action='store_true',
                       help='Append an </s> token to the end of a sentence.')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='sentencepeice tokenizer model.')
    args = parser.parse_args()
    args.rank = 0
    args.make_vocab_size_divisible_by = 1#128
    args.tensor_model_parallel_size = 1
    tokenizer = build_tokenizer(args)
    check_tokenizer(tokenizer)
