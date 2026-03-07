# Copyright (c) 2026 yyang. All rights reserved.
import tokenizer
import dataloader
from torch.nn import functional
import torch
import glob
import time
import sys

isTraining = True


def createModelConfig():
    config = {
        "dimEmb": 384,
        "numLayer": 8,
        "numHead": 6,
        "maxWindowSize": 512,
        "dropoutRate": 0.3,
        "learningRate": 3e-4,
        "numEpoch": 1,
        "batchSize": 16,
        "trainDataRatio": 0.98,
        "temperature": 0.9,
        "topP": 0.9,
    }
    return config


class Normalization:
    def __init__(self, config):
        self.norm = torch.nn.RMSNorm(config["dimEmb"])

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
    def __init__(self, config, cos, sin):
        dimEmb = config["dimEmb"]
        self.numHead = config["numHead"]
        # Use Kaiming initialization for better convergence
        self.wQuery = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wKey = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wValue = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.wOut = torch.nn.Linear(dimEmb, dimEmb, bias=False)
        self.dropout = torch.nn.Dropout(config["dropoutRate"])
        self.cos, self.sin = cos, sin

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
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)

    def applyRoPE(self, q, k, inputLen):
        # q and k are (batchSize, numHead, inputLen, dimHead)
        # cos and sin are (inputLen, dimHead//2)
        cos, sin = self.cos[:inputLen, :], self.sin[:inputLen, :]
        # cos and sin are (1, 1, inputLen, dimHead//2)
        # now they are matched with Q and K
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        # qeven, qodd are (batchSize, numHead, inputLen, dimHead//2)
        # where last dimension is [q0,q2,q4...] [q1,q3,q5...]
        qeven, qodd = q[..., ::2], q[..., 1::2]
        keven, kodd = k[..., ::2], k[..., 1::2]
        # q0*cos(θ) - q1*sin(θ)
        # q1*cos(θ) + q0*sin(θ)
        # ... and so on
        rotatedQeven = qeven * cos - qodd * sin
        rotatedQodd = qodd * cos + qeven * sin
        rotatedKeven = keven * cos - kodd * sin
        rotatedKodd = kodd * cos + keven * sin
        # rotatedQ and rotatedK are (batchSize, numHead, inputLen, dimHead//2, 2)
        # so I should flatten the last dimension to get back to
        # (batchSize, numHead, inputLen, dimHead)
        rotatedQ = torch.stack([rotatedQeven, rotatedQodd], dim=-1).flatten(-2)
        rotatedK = torch.stack([rotatedKeven, rotatedKodd], dim=-1).flatten(-2)
        return rotatedQ, rotatedK

    def compute(self, x):
        # compute Q,K,V at once, they are in shape of [batchSize, dimEmb, dimEmb]
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
        queries = query.view(batchSize, inputLen,
                             self.numHead, dimHead).transpose(1, 2)
        keys = key.view(batchSize, inputLen, self.numHead,
                        dimHead).transpose(1, 2)
        values = value.view(batchSize, inputLen,
                            self.numHead, dimHead).transpose(1, 2)
        # use RoPE to understand relative position of tokens
        queries, keys = self.applyRoPE(queries, keys, inputLen)
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
        out = out.transpose(1, 2).contiguous().view(
            batchSize, inputLen, dimEmb)
        return self.wOut(out)


class Transformer:
    def __init__(self, config, cos, sin):
        self.attn = Attention(config, cos, sin)
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


class Model:
    def __init__(self, config, vocabSize):
        torch.manual_seed(0xCAFEBABE)
        dimEmb = config["dimEmb"]
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.vocabSize = vocabSize
        self.tokenEmbedding = torch.nn.Embedding(vocabSize, dimEmb)
        dimHead = dimEmb // config["numHead"]
        cos, sin = self.initRoPE(config["maxWindowSize"], dimHead)
        self.transformers = [
            Transformer(config, cos, sin) for _ in range(config["numLayer"])
        ]
        self.finalNorm = Normalization(config)
        self.out = torch.nn.Linear(dimEmb, vocabSize, bias=False)
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=config["learningRate"])

    def parameters(self):
        params = list(self.tokenEmbedding.parameters())
        for t in self.transformers:
            params += t.parameters()
        params += self.finalNorm.parameters()
        params += list(self.out.parameters())
        return params

    def to(self, device):
        self.device = device
        self.tokenEmbedding.to(device)
        for t in self.transformers:
            t.to(device)
        self.finalNorm.to(device)
        self.out.to(device)

    def initRoPE(self, maxWindowSize, dimHead):
        # freq = 10000 ^ (-2 * i / dimHead), where i is in [0, 1,..., dimHead//2]
        i = torch.arange(start=0, end=dimHead // 2, device=self.device)
        freq = 10000.0 ** (-2 * i / dimHead)
        pos = torch.arange(maxWindowSize, device=self.device)
        theta = torch.outer(pos, freq)
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        return cos, sin

    def compute(self, input):
        input = input.to(self.device)
        # Transformer moment
        x = self.tokenEmbedding(input)
        for transformer in self.transformers:
            x = transformer.compute(x)
        x = self.finalNorm.compute(x)
        return self.out(x)

    def loss(self, output, target, backward=True):
        target = target.to(self.device)
        # cross-entrypy loss asks for (numSample, numClass) and (numSample) as input
        # it means every sample has a prob distribution over all classes as output
        # and a single class as target
        # while I have out(batchSize, inputLen(numSample), vocabSize(numClass))
        # and target(batchSize, inputLen(numSample)), so I need to flatten them
        # as out(batchSize * inputLen, vocabSize) and target(batchSize * inputLen)
        output = output.view(
            output.shape[0] * output.shape[1], output.shape[2])
        target = target.view(target.shape[0] * target.shape[1])
        loss = functional.cross_entropy(output, target)
        lossVal = loss.item()
        # backward pass to update weights
        if backward:
            loss.backward()
            # prevent the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return lossVal

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
        print(f"@@ Model Configuration:")
        print(f"@@    Device: {self.device}")
        print(f"@@    Model Parameters: {totalParams}")
        print(f"@@    Model Config: {self.config}")
        print(f"@@    VocabSize: {self.vocabSize}")

    def topP(self, logits):
        logits, idx = torch.sort(logits, descending=True)
        # [0.5,0.3,0.1,0.1]
        probs = torch.softmax(logits, dim=-1)
        # [0.5,0.8,0.9,1.0] if topP=0.85
        cum = torch.cumsum(probs, dim=-1)
        # [False, False, True, True]
        removeMask = cum > self.config["topP"]
        # keep the first token that makes cumulative probability exceed topP.
        # e.g., keep 0.1 so (0.5+0.3+0.1) >= 0.85
        removeMask[1:] = removeMask[:-1].clone()
        # keep at least one token in case of all tokens are removed
        removeMask[0] = False
        masked = logits.masked_fill(removeMask, -torch.inf)
        filtered = logits.clone()
        filtered.fill_(-torch.inf)
        filtered.scatter_(dim=-1, index=idx, src=masked)
        return filtered

    @torch.no_grad()
    def nextToken(self, tokens):
        t = torch.tensor(tokens, dtype=torch.long, device=self.device)
        logits = self.compute(torch.stack([t]))
        # first batch, last tokens, all logits
        logits = logits[0, -1, :] / self.config["temperature"]
        logits = self.topP(logits)
        probs = torch.softmax(logits, dim=-1)
        nextTokenId = torch.multinomial(probs, num_samples=1)
        return nextTokenId.item()


class SmallGPT:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tokenizer.HuggingFaceTokenizer()
        self.model = Model(config, vocabSize=self.tokenizer.vocabSize())
        # use simple data loader to load Jinyong's novels all at once
        # it should be replaced with large data loader for streaming
        # files = glob.glob("data/pretrain/*.txt")
        # self.dataloader = dataloader.SimpleDataLoader(
        #     self.config, self.tokenizer, files)
        files = glob.glob("data/more/*.txt")
        trainDataRatio = config["trainDataRatio"]
        trainFiles = files[:int(len(files)*trainDataRatio)]
        valFiles = files[int(len(files)*trainDataRatio):]
        self.dataloader = dataloader.LargeDataLoader(config,self.tokenizer,trainFiles,valFiles)

    def validate(self):
        global isTraining
        isTraining = False
        totalLoss = 0.0
        totalBatch = 0
        for (input, target) in self.dataloader.nextValBatch():
            # compute loss without updating weights
            output = self.model.compute(input)
            loss = self.model.loss(output, target, backward=False)
            totalLoss += loss
            totalBatch += 1
        return totalLoss / totalBatch

    def train(self, modelPath="smallgpt.bin"):
        for epoch in range(self.config["numEpoch"]):
            start = time.time()
            totalLoss = 0.0
            totalBatch = 0
            for (input, target) in self.dataloader.nextTrainBatch():
                global isTraining
                isTraining = True
                output = self.model.compute(input)
                loss = self.model.loss(output, target, backward=True)
                totalLoss += loss
                totalBatch += 1
                if totalBatch % 100 == 0:
                    avgValLoss = self.validate()
                    print(
                        f"@@ Progress: {self.dataloader.progress():.2f}% Batch: {totalBatch} TrainLoss: {totalLoss/totalBatch:.4f} ValLoss: {avgValLoss:.4f}")
            end = time.time()
            self.model.saveWeights(modelPath)
            print(f"@@ Epoch: {epoch} Elapsed: {end-start:.2f}s")

    def tuning(self):
        self.config["learningRate"] = 3e-5
        self.model.printConfig()
        self.dataloader = dataloader.SFTDataLoader(self.config, self.tokenizer)
        self.train("smallgpt_tuning.bin")

    @torch.no_grad()
    def predict(self, sentences, maxTokens=30):
        global isTraining
        isTraining = False
        self.model.loadWeights("smallgpt.bin")
        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence)
            for _ in range(maxTokens):
                tokens = tokens[-self.config["maxWindowSize"]:]
                tokens.append(self.model.nextToken(tokens))
            output = self.tokenizer.decode(tokens)
            output = output[len(sentence):]
            print(f"@@ Predict: {sentence}[{output}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        print(f"@@ SmallGPT Mode: {mode}")
        g = SmallGPT(createModelConfig())
        if mode == "train":
            g.train()
        elif mode == "predict":
            sentences = ["杨过和小龙女在", "神雕大侠", "韦小宝和双儿", "围攻光明顶",
                         "郭靖和黄蓉", "张无忌", "令狐冲说", "华山论剑", "桃花岛上", "少林寺", "降龙十八掌"]
            g.predict(sentences)
        elif mode == "tuning":
            g.tuning()
    else:
        print("Usage: python smallgpt.py <train|predict|tuning>")
