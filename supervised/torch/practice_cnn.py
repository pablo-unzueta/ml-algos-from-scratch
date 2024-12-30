import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from tqdm import tqdm


def get_dataloaders(batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_data = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    train_data, val_data = random_split(train_data, lengths=[0.8, 0.2])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list[int] = [8, 16],
        num_classes: int = 10,
        input_size: int = 28,
    ):
        super().__init__()

        channels = [in_channels] + hidden_channels

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                )
                for i in range(len(channels) - 1)
            ]
        )
        feature_size = input_size // (2 * len(self.conv_blocks))  # (W - K + 2P / S) + 1
        self.classifier = nn.Linear(
            channels[-1] * feature_size * feature_size, num_classes
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    num_epochs: int,
    device: str,
) -> None:
    best_val_acc = float("-inf")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, targets in tqdm(train_loader):
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            loss = F.cross_entropy(scores, targets)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        model.train()  # when calling evaluate, model gets put in eval mode. reset it
        scheduler.step(metrics=val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch: [{epoch+1}/{num_epochs}], Train Avg. Loss:{epoch_loss/len(train_loader):.4f}, Val. Acc: {best_val_acc:.2f}"
        )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct_preds = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)

            predictions = model(data).max(1)[1]
            correct_preds += (predictions == targets).sum()

    return 100 * correct_preds / len(loader.dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 3

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        min_lr=1e-6,
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
    )
    test_acc = evaluate(model, test_loader, device=device)
    print(f"Final Test Accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    main()
