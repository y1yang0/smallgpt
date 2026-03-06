from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
import glob, os, sys

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
trainer = BpeTrainer(
    vocab_size=20000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<|endofbook|>"],
)


def trainTokenizer():
    dataDir = "data/pretrain"
    files = glob.glob(os.path.join(dataDir, "*.txt"))
    print(f"Training tokenizer with {files} files")
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
