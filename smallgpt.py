import tiktoken
import torch
import sys

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
    "maxWindowSize": 512,
    "dropoutRate": 0.0,
    "learningRate": 3e-4,
    "numEpoch": 1,
}


class Normalization:
    def __init__(self, dimEmb):
        self.norm = torch.nn.LayerNorm(dimEmb)

    def compute(self, x):
        return self.norm(x)

    def to(self, device):
        self.norm.to(device)

    def parameters(self):
        return list(self.norm.parameters())


class FeedForward:
    def __init__(self, dimEmb):
        # up-project the input to a higher dimension so that the model can find
        # similar patterns in the data, non-linear activation function could
        # "turn on" or "turn off" certain patterns
        self.layer1 = torch.nn.Linear(dimEmb, dimEmb * 4)
        self.layer2 = torch.nn.GELU()
        # down-project the input back to original dimension so that the model
        # can extract the detailed knowledge from the matched patterns
        self.layer3 = torch.nn.Linear(dimEmb * 4, dimEmb)

    def compute(self, x):
        # FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
        return self.layer3(self.layer2(self.layer1(x)))

    def to(self, device):
        self.layer1.to(device)
        self.layer3.to(device)

    def parameters(self):
        return list(self.layer1.parameters()) + list(self.layer3.parameters())


class Attention:
    def __init__(self, dimEmb, dimOut, dropoutRate=config["dropoutRate"]):
        self.dimOut = dimOut
        scale = dimEmb**-0.5
        self.wQuery = torch.nn.Parameter(
            torch.randn(dimEmb, dimOut) * scale, requires_grad=True
        )
        self.wKey = torch.nn.Parameter(
            torch.randn(dimEmb, dimOut) * scale, requires_grad=True
        )
        self.wValue = torch.nn.Parameter(
            torch.randn(dimEmb, dimOut) * scale, requires_grad=True
        )
        self.dropout = torch.nn.Dropout(dropoutRate)

    def parameters(self):
        return [self.wQuery, self.wKey, self.wValue]

    def to(self, device):
        self.wQuery.data = self.wQuery.data.to(device)
        self.wKey.data = self.wKey.data.to(device)
        self.wValue.data = self.wValue.data.to(device)

    def compute(self, x):
        query = x @ self.wQuery
        key = x @ self.wKey
        value = x @ self.wValue
        # attention socre means which tokens are most relevant to current token
        # Attention(Q,K,V) = softmax(mask(Q@K^T / sqrt(d_k))) @ V
        mask = torch.tril(torch.ones(query.shape[0], query.shape[0], device=x.device))
        attnScore = query @ key.T / (key.shape[-1] ** 0.5)
        # use causal mask to prevent the current token from seeing future tokens
        attnScore = attnScore.masked_fill(mask == 0, -torch.inf)
        # apply softmax to get the attention weights
        attnWeights = torch.softmax(attnScore, dim=-1)
        # apply dropout to prevent overfitting
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
    def __init__(self, dimEmb, maxWindowSize, numLayer, dropoutRate, learningRate):
        torch.manual_seed(0xCAFEBABE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.maxWindowSize = maxWindowSize
        self.tokenEmbedding = torch.nn.Embedding(self.tokenizer.n_vocab, dimEmb)
        self.posEmbedding = torch.nn.Embedding(maxWindowSize, dimEmb)
        self.transformers = [Transformer(dimEmb, dropoutRate) for _ in range(numLayer)]
        self.finalNorm = Normalization(dimEmb)
        self.out = torch.nn.Linear(dimEmb, self.tokenizer.n_vocab, bias=False)
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learningRate)

    def encode(self, input):
        return self.tokenizer.encode(input)

    def decode(self, input):
        return self.tokenizer.decode(input)

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
        pos = torch.arange(len(input), device=self.device)
        x = self.tokenEmbedding(input) + self.posEmbedding(pos)
        for transformer in self.transformers:
            x = transformer.compute(x)
        x = self.finalNorm.compute(x)
        return self.out(x)

    def save(self, path):
        torch.save([p.data.cpu() for p in self.parameters()], path)
        print(f"@@ Model saved to {path}")

    def load(self, path):
        state = torch.load(path, weights_only=True, map_location=self.device)
        for p, data in zip(self.parameters(), state):
            p.data.copy_(data)
        print(f"@@ Model loaded from {path}")

    def printConfig(self):
        totalParams = sum(p.numel() for p in self.parameters())
        print(f"@@ SmallGPT Configuration:")
        print(f"@@    Device: {self.device}")
        print(f"@@    Total Parameters: {totalParams}")
        print(f"@@    Memory Usage: {totalParams * 4 / 1024 / 1024:.2f} MB")

    def nextToken(self, input, temperature=0.9):
        with torch.no_grad():
            logits = self.compute(input)
            logits = logits[-1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nextTokenId = torch.multinomial(probs, num_samples=1)
            return nextTokenId.item()

    def loadDataset(self):
        dataset = []
        for path in config["dataset"]:
            with open(path, "r", encoding="utf-8") as f:
                tokens = torch.tensor(self.encode(f.read()))   
                for i in range(0, len(tokens) - 1, self.maxWindowSize):
                    chunk = tokens[i : i + self.maxWindowSize + 1]
                    dataset.append((chunk[:-1], chunk[1:]))
        return dataset

    def train(self, numEpoch):
        # read training data from files
        dataset = self.loadDataset()
        print(f"@@ Loaded dataset size: {len(dataset)}")
        # train the model
        for epoch in range(numEpoch):
            for idx, (input, target) in enumerate(dataset):
                input, target = input.to(self.device), target.to(self.device)
                output = self.compute(input)
                loss = torch.nn.functional.cross_entropy(output, target)
                print(
                    f"\r@@ Epoch: {epoch} Progress: {idx/len(dataset)*100:.2f}% Loss: {loss.item():.4f}",
                    end="",
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            # save the model weights
            self.save("smallgpt.bin")

    def predict(self, text, maxTokens=30):
        print(f"@@ Input: {text}")
        tokenIds = self.encode(text)
        with torch.no_grad():
            for _ in range(maxTokens):
                window = tokenIds[-self.maxWindowSize :]
                t = self.nextToken(torch.tensor(window, dtype=torch.long, device=self.device))
                tokenIds.append(t)
        print(f"@@ Output: {self.decode(tokenIds)}")


smallGPT = SmallGPT(
    config["dimEmb"],
    config["maxWindowSize"],
    config["numLayer"],
    config["dropoutRate"],
    config["learningRate"],
)

if len(sys.argv) > 1 and sys.argv[1] == "train":
    smallGPT.printConfig()
    smallGPT.train(config["numEpoch"])
else:
    smallGPT.load("smallgpt.bin")
    smallGPT.predict("杨过和小龙女在")
    smallGPT.predict("神雕大侠")
    smallGPT.predict("韦小宝和双儿")
    smallGPT.predict("围攻光明顶")
