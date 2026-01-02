import tiktoken
import torch

config = {
    "dimEmb": 1024,
    "numLayer": 2,
    "maxWindowSize": 512,
    "dropoutRate": 0.0,
    "learningRate": 1e-2,
    "numEpoch": 10,
}


class Normalization:
    def __init__(self, dimEmb):
        self.norm = torch.nn.LayerNorm(dimEmb)

    def compute(self, x):
        return self.norm(x)

    def parameters(self):
        return list(self.norm.parameters())


class FeedForward:
    def __init__(self, dimEmb):
        self.layer1 = torch.nn.Linear(dimEmb, dimEmb * 4)
        self.layer2 = torch.nn.GELU()
        self.layer3 = torch.nn.Linear(dimEmb * 4, dimEmb)

    def compute(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

    def parameters(self):
        return list(self.layer1.parameters()) + list(self.layer3.parameters())


class Attention:
    def __init__(self, dimEmb, dimOut, dropoutRate=config["dropoutRate"]):
        self.dimOut = dimOut
        self.wQuery = torch.nn.Parameter(
            torch.randn(dimEmb, dimOut), requires_grad=True
        )
        self.wKey = torch.nn.Parameter(torch.randn(dimEmb, dimOut), requires_grad=True)
        self.wValue = torch.nn.Parameter(
            torch.randn(dimEmb, dimOut), requires_grad=True
        )
        self.dropout = torch.nn.Dropout(dropoutRate)

    def parameters(self):
        return [self.wQuery, self.wKey, self.wValue]

    def compute(self, x):
        query = x @ self.wQuery
        key = x @ self.wKey
        value = x @ self.wValue
        # Attention(Q,K,V) = softmax(mask(Q@K^T / sqrt(d_k))) @ V
        mask = torch.tril(torch.ones(query.shape[0], query.shape[0]))
        attnScore = query @ key.T / (key.shape[-1] ** 0.5)
        # causal mask to prevent the current token from seeing future tokens
        attnScore = attnScore.masked_fill(mask == 0, -torch.inf)
        attnWeights = torch.softmax(attnScore, dim=-1)
        # dropout to prevent overfitting
        attnWeights = self.dropout(attnWeights)
        return attnWeights @ value


class Transformer:
    def __init__(self, dimEmb, dropoutRate):
        self.attn = Attention(dimEmb, dimEmb, dropoutRate)
        self.norm1 = Normalization(dimEmb)
        self.norm2 = Normalization(dimEmb)
        self.ffn = FeedForward(dimEmb)

    def compute(self, x):
        x = x + self.attn.compute(self.norm1.compute(x))
        x = x + self.ffn.compute(self.norm2.compute(x))
        return x

    def parameters(self):
        return (
            self.attn.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
        )


class SmallGPT:
    def __init__(self, dimEmb, maxWindowSize, numLayer, dropoutRate):
        torch.manual_seed(0xCAFEBABE)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenEmbedding = torch.nn.Embedding(self.tokenizer.n_vocab, dimEmb)
        self.posEmbedding = torch.nn.Embedding(maxWindowSize, dimEmb)
        self.transformers = [Transformer(dimEmb, dropoutRate) for _ in range(numLayer)]
        self.finalNorm = Normalization(dimEmb)
        self.out = torch.nn.Linear(dimEmb, self.tokenizer.n_vocab, bias=False)

    def tokenize(self, input):
        return torch.tensor(self.tokenizer.encode(input))

    def parameters(self):
        params = list(self.tokenEmbedding.parameters()) + list(
            self.posEmbedding.parameters()
        )
        for t in self.transformers:
            params += t.parameters()
        params += self.finalNorm.parameters()
        params += list(self.out.parameters())
        return params

    def compute(self, input):
        # attach the token embeddings with the position sequence [0,1,2,...]
        x = self.tokenEmbedding(input) + self.posEmbedding(torch.arange(len(input)))
        for transformer in self.transformers:
            x = transformer.compute(x)
        x = self.finalNorm.compute(x)
        return self.out(x)

    def clearGrad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def updateWeight(self, learningRate=1e-4):
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p.data -= learningRate * p.grad
        self.clearGrad()

    def printConfig(self):
        totalParams = sum(p.numel() for p in self.parameters())
        print(f"@@ SmallGPT Configuration:")
        print(f"@@    Total Parameters: {totalParams}")
        print(f"@@    Memory Usage: {totalParams * 4 / 1024 / 1024:.2f} MB")

    def nextToken(self, input):
        with torch.no_grad():
            logits = self.compute(input)
        nextTokenId = logits[-1, :].argmax(dim=-1)
        return self.tokenizer.decode([nextTokenId.item()])

    def train(self, txts, learningRate, numEpoch):
        dataset = []
        for txt in txts:
            tokens = self.tokenize(txt)
            dataset.append((tokens[:-1], tokens[1:]))

        for epoch in range(numEpoch):
            for input, target in dataset:
                output = self.compute(input)
                loss = torch.nn.functional.cross_entropy(output, target)
                print(f"@@ Epoch {epoch}: Loss: {loss.item()}")
                loss.backward()
                self.updateWeight(learningRate)


smallGPT = SmallGPT(
    config["dimEmb"], config["maxWindowSize"], config["numLayer"], config["dropoutRate"]
)
smallGPT.printConfig()

txts = [
    "The color of apple is red",
    "The color of banana is yellow",
    "The color of cherry is red",
    "The color of pineapple is yellow",
    "The color of strawberry is red",
]

smallGPT.train(txts, config["learningRate"], config["numEpoch"])

input = "The color of cherry is"
print(f"@@ Input: {input}")
print(f"@@ Next Token: {smallGPT.nextToken(smallGPT.tokenize(input))}")
