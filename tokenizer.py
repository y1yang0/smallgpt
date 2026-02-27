from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation, Sequence
import sys

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
trainer = BpeTrainer(
    vocab_size=20000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)


def trainTokenizer():
    files = [
        "data/金庸-书剑恩仇录txt精校版.txt",
        "data/金庸-侠客行txt全本精校版.txt",
        "data/金庸-倚天屠龙记txt精校版.txt",
        "data/金庸-天龙八部txt精校版.txt",
        "data/金庸-射雕英雄传txt精校版.txt",
        "data/金庸-白马啸西风txt精校版.txt",
        "data/金庸-碧血剑txt精校版.txt",
        "data/金庸-神雕侠侣txt精校版.txt",
        "data/金庸-笑傲江湖txt精校版.txt",
        "data/金庸-越女剑txt精校版.txt",
        "data/金庸-连城诀txt精校版.txt",
        "data/金庸-雪山飞狐txt精校版.txt",
        "data/金庸-飞狐外传txt精校版.txt",
        "data/金庸-鸳鸯刀txt精校版.txt",
        "data/金庸-鹿鼎记txt精校版.txt",
    ]
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
