# Copyright (c) 2026 yyang. All rights reserved.
from tokenizers import Tokenizer
from torch.nn import functional as functional
import tiktoken
import torch
import sys
import random

config = {
    "dataset": [
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
    ],
    "dimEmb": 512,
    "numLayer": 8,
    "numHead": 8,
    "maxWindowSize": 512,
    "dropoutRate": 0.1,
    "learningRate": 3e-4,
    "numEpoch": 10,
    "batchSize": 16,
}

isTraining = True

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
        return self.tokenizer.decode(ids)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def vocabSize(self):
        return self.tokenizer.get_vocab_size()

class DataLoader:
    def __init__(self, config, tokenizer):
        self.maxWindowSize = config["maxWindowSize"]
        self.batchSize = config["batchSize"]
        self.tokenizer = tokenizer
        self.numTokens = 0
        self.batches = self.loadDataBatch()

    def loadDataBatch(self):
        # all books concatenated into a single string and split then into chunks
        # of maxWindowSize, each chunk is a (input, target) pair
        dataset = []
        for path in config["dataset"]:
            with open(path, "r", encoding="utf-8") as f:
                tokens = torch.tensor(self.tokenizer.encode(f.read()))
                self.numTokens += len(tokens)
                for i in range(0, len(tokens) - 1, self.maxWindowSize):
                    chunk = tokens[i : i + self.maxWindowSize + 1]
                    if len(chunk) != self.maxWindowSize + 1:
                        continue  # drop the last unaligned chunk
                    dataset.append((chunk[:-1], chunk[1:]))

        # pack the dataset into smaller batches, i.e.
        # [(input, target), (input1, target1), ...] =>
        # batch1: [input, input1, ...], [target, target1, ...]
        # batch2: [inputN, inputN+1, ...], [targetN, targetN+1, ...]
        batches = []
        for idx in range(0, len(dataset), self.batchSize):
            # [(input, target), (input1, target1), ...]
            batch = dataset[idx : idx + self.batchSize]
            # [input, input1, ...], [target, target1, ...]
            inputBatch, targetBatch = zip(*batch)
            # tensor([input, input1, ...]), tensor([target, target1, ...])
            inputBatch = torch.stack(inputBatch)
            targetBatch = torch.stack(targetBatch)
            batches.append((inputBatch, targetBatch))
        return batches

    def numBatches(self):
        return len(self.batches)

    def __iter__(self):
        # shuffle the dataset every epoch to prevent model from being overfitted
        random.shuffle(dataset)
        return iter(self.batches)

class Normalization:
    def __init__(self, config):
        self.norm = torch.nn.LayerNorm(config["dimEmb"])

    def compute(self, x):
        return self.norm(x)

    def to(self, device):
        self.norm.to(device)

    def parameters(self):
        return list(self.norm.parameters())


class FeedForward:
    def __init__(self, config):
        dimEmb = config["dimEmb"]
        dimHidden = int(2 / 3 * 4 * dimEmb)
        self.wGate = torch.nn.Linear(dimEmb, dimHidden, bias=False)
        self.wValue = torch.nn.Linear(dimEmb, dimHidden, bias=False)
        self.wOut = torch.nn.Linear(dimHidden, dimEmb, bias=False)
        self.dropout = torch.nn.Dropout(config["dropoutRate"])

    def compute(self, x):
        # SwiGLU(x) = (SiLU(x @ wGate) * x @ wValue) @ wOut
        # SiLU(x @ wGate) computes the 0~1 gate value to control how much
        # features from (x @ wValue) should be extracted and wOut projects
        # weighted features to real knowledge
        x = functional.silu(self.wGate(x)) * self.wValue(x)
        if isTraining:
            x = self.dropout(x)
        return self.wOut(x)

    def to(self, device):
        self.wGate.to(device)
        self.wValue.to(device)
        self.wOut.to(device)
        self.dropout.to(device)

    def parameters(self):
        return (
            list(self.wGate.parameters())
            + list(self.wValue.parameters())
            + list(self.wOut.parameters())
        )


class Attention:
    def __init__(self, config):
        dimEmb = config["dimEmb"]
        self.numHead = config["numHead"]
        # Use Kaiming initialization for better convergence
        self.wQuery = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wKey = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wValue = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wOut = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.dropout = torch.nn.Dropout(config["dropoutRate"])

    def parameters(self):
        return (
            list(self.wQuery.parameters())
            + list(self.wKey.parameters())
            + list(self.wValue.parameters())
            + list(self.wOut.parameters())
        )

    def to(self, device):
        self.wQuery.to(device)
        self.wKey.to(device)
        self.wValue.to(device)
        self.wOut.to(device)
        self.dropout.to(device)

    def compute(self, x):
        # compute Q,K,V at once
        query = self.wQuery(x)
        key = self.wKey(x)
        value = self.wValue(x)
        # split the Q,K,V tensor into multiple heads, each head has dimHead
        # dimensions. Intuitively, I view old [batchSize, inputLen, dimEmb] as
        # [batchSize, numHead, inputLen, dimHead], but it turns out that it
        # should be firstly viewed as [batchSize, inputLen, numHead, dimHead]
        # and transpose(1,2) dimensions to get the desired shape
        batchSize, inputLen, dimEmb = x.shape
        dimHead = dimEmb // self.numHead
        queries = query.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        keys = key.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        values = value.view(batchSize, inputLen, self.numHead, dimHead).transpose(1, 2)
        # compute Attention(Q,K,V) = softmax(mask(Q@K^T / sqrt(d_k))) @ V
        #
        # attention socre means which tokens are most relevant to current token
        #   Q(batchSize, numHead, inputLen, dimHead) @ K^T(batchSize, numHead, dimHead, inputLen)
        #   = attnScore(batchSize, numHead, inputLen, inputLen)
        attnScore = queries @ keys.transpose(-2, -1) / (dimHead**0.5)
        # use causal mask to prevent the current token from seeing future tokens
        #   attnScore(batchSize, numHead, inputLen, inputLen) @ mask(batchSize, numHead, inputLen, inputLen)
        #   = maskedAttnScore(batchSize, numHead, inputLen, inputLen)
        mask = torch.tril(torch.ones(inputLen, inputLen, device=x.device))
        attnScore = attnScore.masked_fill(mask == 0, -torch.inf)
        # apply softmax to get the attention weights
        attnWeights = torch.softmax(attnScore, dim=-1)
        # apply dropout to prevent overfitting
        if isTraining:
            attnWeights = self.dropout(attnWeights)
        # apply weights to the values to get the output
        #   attnWeights(batchSize, numHead, inputLen, inputLen) @ V(batchSize, numHead, inputLen, dimHead)
        #   = out(batchSize, numHead, inputLen, dimHead)
        out = attnWeights @ values
        # merge all attention heads back and apply final projection to understand
        # how to combine the information from all heads
        #   out(batchSize, numHead, inputLen, dimHead)
        #   = out(batchSize, inputLen, dimEmb)
        out = out.transpose(1, 2).contiguous().view(batchSize, inputLen, dimEmb)
        return self.wOut(out)


class Transformer:
    def __init__(self, config):
        self.attn = Attention(config)
        self.norm1 = Normalization(config)
        self.norm2 = Normalization(config)
        self.ffn = FeedForward(config)

    def compute(self, x):
        x = x + self.attn.compute(self.norm1.compute(x))
        x = x + self.ffn.compute(self.norm2.compute(x))
        return x

    def to(self, device):
        self.attn.to(device)
        self.norm1.to(device)
        self.norm2.to(device)
        self.ffn.to(device)

    def parameters(self):
        return (
            self.attn.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
        )


class SmallGPT:
    def __init__(self, config):
        torch.manual_seed(0xCAFEBABE)
        dimEmb = config["dimEmb"]
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = HuggingFaceTokenizer()
        self.tokenEmbedding = torch.nn.Embedding(self.tokenizer.vocabSize(), dimEmb)
        self.posEmbedding = torch.nn.Embedding(config["maxWindowSize"], dimEmb)
        self.transformers = [Transformer(config) for _ in range(config["numLayer"])]
        self.finalNorm = Normalization(config)
        self.out = torch.nn.Linear(dimEmb, self.tokenizer.vocabSize(), bias=False)
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config["learningRate"])
        self.dataloader = DataLoader(config, tokenizer=self.tokenizer)

    def parameters(self):
        params = list(self.tokenEmbedding.parameters()) + list(
            self.posEmbedding.parameters()
        )
        for t in self.transformers:
            params += t.parameters()
        params += self.finalNorm.parameters()
        params += list(self.out.parameters())
        return params

    def to(self, device):
        self.device = device
        self.tokenEmbedding.to(device)
        self.posEmbedding.to(device)
        for t in self.transformers:
            t.to(device)
        self.finalNorm.to(device)
        self.out.to(device)

    def compute(self, input):
        # attach the token embeddings with the position sequence [0,1,2,...]
        pos = torch.arange(input.shape[1], device=self.device)
        x = self.tokenEmbedding(input) + self.posEmbedding(pos)
        for transformer in self.transformers:
            x = transformer.compute(x)
        x = self.finalNorm.compute(x)
        return self.out(x)

    def saveWeights(self, path):
        torch.save([p.data.cpu() for p in self.parameters()], path)
        print(f"@@ Model saved to {path}")

    def loadWeights(self, path):
        state = torch.load(path, weights_only=True, map_location=self.device)
        for p, data in zip(self.parameters(), state):
            p.data.copy_(data)
        print(f"@@ Model loaded from {path}")

    def printConfig(self):
        totalParams = sum(p.numel() for p in self.parameters())
        print(f"@@ SmallGPT Configuration:")
        print(f"@@    Device: {self.device}")
        print(f"@@    Model Parameters: {totalParams}")
        print(f"@@    Model Memory Usage: {totalParams * 4 / 1024 / 1024:.2f} MB")
        print(f"@@    Tokenizer: {self.tokenizer.__class__.__name__}")
        print(f"@@    Tokenizer VocabSize: {self.tokenizer.vocabSize()}")
        print(f"@@    Dataset Batches: {self.dataloader.numBatches()}")
        print(f"@@    Dataset Tokens: {self.dataloader.numTokens}")
        print(f"@@    Dataset WindowSize: {self.dataloader.maxWindowSize}")

    def nextToken(self, input, temperature=0.9):
        with torch.no_grad():
            logits = self.compute(torch.stack([input]))
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nextTokenId = torch.multinomial(probs, num_samples=1)
            return nextTokenId.item()

    def train(self, numEpoch):
        global isTraining
        isTraining = True
        for epoch in range(numEpoch):
            for (idx, (input, target)) in enumerate(self.dataloader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.compute(input)
                # cross-entrypy loss asks for (numSample, numClass) and (numSample) as input
                # it means every sample has a prob distribution over all classes as output
                # and a single class as target
                # while I have out(batchSize, inputLen(numSample), vocabSize(numClass))
                # and target(batchSize, inputLen(numSample)), so I need to flatten them
                # as out(batchSize * inputLen, vocabSize) and target(batchSize * inputLen)
                output = output.view(output.shape[0] * output.shape[1], output.shape[2])
                target = target.view(target.shape[0] * target.shape[1])
                loss = functional.cross_entropy(output, target)
                print(
                    f"\r@@ Epoch: {epoch} Progress: {idx/self.dataloader.numBatches()*100:.2f}% Loss: {loss.item():.4f}",
                    end="",
                )
                loss.backward()
                # prevent the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            # save the model weights
            self.saveWeights("smallgpt.bin")
            smallGPT.predict("杨过和小龙女在")

    def predict(self, text, maxTokens=30):
        global isTraining
        isTraining = False
        print(f"@@ Input: {text}")
        tokenIds = self.tokenizer.encode(text)
        with torch.no_grad():
            for _ in range(maxTokens):
                window = tokenIds[-self.maxWindowSize() :]
                t = self.nextToken(
                    torch.tensor(window, dtype=torch.long, device=self.device)
                )
                tokenIds.append(t)
        print(f"@@ Output: {self.tokenizer.decode(tokenIds)}")


smallGPT = SmallGPT(config)

if len(sys.argv) > 1 and sys.argv[1] == "train":
    smallGPT.printConfig()
    smallGPT.train(config["numEpoch"])
else:
    smallGPT.loadWeights("smallgpt.bin")
    smallGPT.predict("杨过和小龙女在")
    smallGPT.predict("神雕大侠")
    smallGPT.predict("韦小宝和双儿")
    smallGPT.predict("围攻光明顶")
    smallGPT.predict("郭靖和黄蓉")
    smallGPT.predict("张无忌")
    smallGPT.predict("令狐冲说")
    smallGPT.predict("华山论剑")
    smallGPT.predict("桃花岛上")
    smallGPT.predict("少林寺")
    smallGPT.predict("降龙十八掌")
