import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import json
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
from tqdm import tqdm


class Tokenizer:
    def __init__(self, data: str, vocab_path: str = None):
        """Character level tokenizer

        Args:
            data (str): Text passage to tokenize
            vocab_path (str, optional): If already tokenized, use this file instead. Defaults to None.

        Raises:
            ValueError: Need to provide data or vocab_path
        """
        if vocab_path and Path(vocab_path).exists():
            vocab = json.load(vocab_path)
            self.stoi = vocab["stoi"]
            self.itos = vocab["itos"]
        elif data:
            chars = list(sorted(set(data)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            # save dict
            vocab_path = Path(__file__).parent / "vocab.json"
            json.dump({"stoi": self.stoi, "itos": self.itos}, open(vocab_path, "w"))
        else:
            raise ValueError("Must provide vocab_path or data")

    def encode(self, s: str) -> list[int]:
        return [self.stoi[char] for char in s]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[i] for i in tokens)


class SPDataset(Dataset):
    def __init__(self, data: str, block_size: int, tokenizer: Tokenizer):
        self.block_size = block_size

        # encode data
        data_ids = torch.tensor(tokenizer.encode(data), dtype=torch.long)

        n = len(data_ids) - block_size
        self.x = torch.stack([data_ids[i : i + block_size] for i in range(n)])
        self.y = torch.stack([data_ids[i + 1 : i + block_size + 1] for i in range(n)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_SP_dataloaders(
    dataset: SPDataset,
    tokenizer: Tokenizer,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = random_split(dataset, lengths=[0.9, 0.1])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


@dataclass
class Config:
    n_embd: int
    n_head: int
    n_blocks: int
    vocab_size: int
    dropout: float
    block_size: int
    lr: float
    bias: bool


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.mlp = MLP(config)

    def forward(self, x):
        x_norm = self.ln1(x)

        # self attention
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)

        x = x + attn_out

        # MLP with res
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(config.vocab_size, config.n_embd),
                pos_emb=nn.Embedding(config.block_size, config.n_embd),
                dropout=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
                ln_f=nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device

        B, T = idx.shape  # batch, seq len

        tok_emb = self.transformer.tok_emb(idx)  # [batch_size, seq_len, n_embd]

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.pos_emb(pos)  # [seq_len, n_embd]

        idx = self.transformer.dropout(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            idx = block(idx)
        idx = self.transformer.ln_f(idx)
        logits = self.lm_head(idx)
        return logits


def train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    device: str,
):
    model.train()
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(tqdm(train_loader)):
            data, targets = data.to(device), targets.to(device)

            logits = model(data)
            # logits [B, T, V] -> [B * T, V]
            # targets [B, T] -> [B * T]
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                print(f"Loss at minibatch: {i+1}: {loss.item()=:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        n_embd=384,
        n_head=6,
        n_blocks=6,
        block_size=256,
        vocab_size=65,
        dropout=0.0,
        lr=0.001,
        bias=False,
    )
    with open("input.txt", "r") as f:
        data = f.read()
    tokenizer = Tokenizer(data)
    dataset = SPDataset(data, config.block_size, tokenizer)
    train_dataloader, val_dataloader = get_SP_dataloaders(
        dataset, tokenizer, batch_size=64
    )
    model = GPT(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    train(
        model=model,
        train_loader=train_dataloader,
        num_epochs=1,
        optimizer=optimizer,
        device=device,
    )


if __name__ == "__main__":
    main()
