import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import requests
from typing import Tuple
import json


class ShakespeareTokenizer:
    def __init__(self, data: str = None, vocab_path: str = None):
        if vocab_path and Path(vocab_path).exists():
            vocab = json.load(open(vocab_path))
            self.stoi = vocab["stoi"]
            self.itos = vocab["itos"]
        elif data:
            chars = sorted(list(set(data)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            # save vocab
            vocab_path = Path(__file__).parent / "vocab.json"
            json.dump({"stoi": self.stoi, "itos": self.itos}, open(vocab_path, "w"))
        else:
            raise ValueError("Either data of vocab_path must be provided...")

        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list | torch.Tensor) -> str:
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        if isinstance(tokens[0], (list, tuple)):
            tokens = tokens[0]

        return "".join(self.itos[i] for i in tokens)


class ShakespeareDataset(Dataset):
    def __init__(self, data: str, tokenizer: ShakespeareTokenizer, block_size: int = 8):
        self.block_size = block_size

        # encode the data
        data_ids = torch.tensor(tokenizer.encode(data), dtype=torch.long)

        # create sequences
        n = len(data_ids) - block_size  # prevents accessing data outside array at end
        self.x = torch.stack([data_ids[i : i + block_size] for i in range(n)])
        self.y = torch.stack([data_ids[i + 1 : i + block_size + 1] for i in range(n)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_shakespeare_data() -> str:
    input_file_path = Path().cwd() / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
        data = f.read()

    return data


def get_shakespeare_dataloaders(
    dataset: ShakespeareDataset,
    tokenizer: ShakespeareTokenizer,
    batch_size: int = 32,
    block_size: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    # split train/val
    train_dataset, val_dataset = random_split(dataset, lengths=[0.9, 0.1])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data = get_shakespeare_data()
    tokenizer = ShakespeareTokenizer(data=data, vocab_path="vocab.json")
    dataset = ShakespeareDataset(data=data, tokenizer=tokenizer, block_size=8)
    train_loader, val_loader = get_shakespeare_dataloaders(
        dataset=dataset, tokenizer=tokenizer, batch_size=64, block_size=8
    )

    for x, y in train_loader:
        print(x[0])
        print(tokenizer.decode(x[0]))
        print(y[0])
        print(tokenizer.decode(y[0]))
        break

    vocab_size = tokenizer.vocab_size
    print(f"{vocab_size=}")
