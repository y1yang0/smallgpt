# Copyright (c) 2026 yyang. All rights reserved.
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
import glob, os, sys

# My first learned tokenizer
class TiktokenTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def decode(self, input):
        return self.tokenizer.decode(input)

    def encode(self, input):
        return self.tokenizer.encode(input)

    def vocabSize(self):
        return self.tokenizer.n_vocab


# Self-trained tokenizer for Jinyong-specific dataset, from huggingface/tokenizer
class HuggingFaceTokenizer:
    def __init__(self):
        path = "data/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(path)

    def decode(self, ids):
        sentence = ""
        for d in ids:
            text = self.tokenizer.decode([d])
            sentence += text
        return sentence

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def vocabSize(self):
        return self.tokenizer.get_vocab_size()

def trainTokenizer():
    dataDir = "data/pretrain"
    files = glob.glob(os.path.join(dataDir, "*.txt"))
    print(f"@@Training tokenizer with {files} files")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
    trainer = BpeTrainer(
        vocab_size=20000,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<|endofbook|>"],
    )

    tokenizer.train(files, trainer)
    path = "data/tokenizer.json"
    tokenizer.save(path)
    print(f"Save tokenizer to {path}")

def testTokenizer():
    path = "data/tokenizer.json"
    tokenizer = Tokenizer.from_file(path)
    sentences = ["杨过和小龙女在古墓。", "神雕大侠，为国为民。", "华山论剑！", "pretty", "黄蓉"]
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        print(f"@@ Encoded: '{encoded.tokens}'")
        decoded_sentence = tokenizer.decode(encoded.ids)
        print(f"@@ Decoded: '{decoded_sentence}'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainTokenizer()
    else:
        testTokenizer()
