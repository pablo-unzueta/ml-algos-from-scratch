import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import json
from pathlib import Path
from typing import Tuple


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


class SPDataLoader:
    def __init__(
        self,
        dataset: SPDataset,
        tokenizer: Tokenizer,
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset, val_dataset = random_split(dataset, lengths=[0.9, 0.1])

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, sampler=)


def main():
    with open("input.txt", "r") as f:
        data = f.read()
    tokenizer = Tokenizer(data)
    dataset = SPDataset(data, 8, tokenizer)


if __name__ == "__main__":
    main()
