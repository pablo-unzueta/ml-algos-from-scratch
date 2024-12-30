import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
from typing import Tuple


class CNNPabloVModuleList(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_channels: list[int] = [8, 16],
        input_size: int = 28,
    ):
        super().__init__()

        channels = [in_channels] + hidden_channels
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        # don't forget that calculating kernel size formula
        # ((W - K + 2P)/S) + 1
        # W: input size
        # K: kernel size
        # P: padding
        # S: stride
        # for maxpool2d in line above:
        # K=2, P=0, S=2
        # ((28 - 2 + 0)/2) + 1 = 14
        # for second maxpool2d
        # ((14 - 2 + 0)/2) + 1 = 7
        feature_size = input_size // (2 ** len(self.conv_layers))
        self.classifer = nn.Linear(
            hidden_channels[-1] * feature_size * feature_size, num_classes
        )

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        x = torch.flatten(x, 1) # flatten starting at dim 1
        # torch.flatten(x, 1) converts it to:
        # [batch_size, channels * height * width]
        # e.g., [64, 16 * 7 * 7] = [64, 784]

        # torch.flatten(x)  # WRONG!
        # Would flatten to [64 * 16 * 7 * 7] = [7840]
        # Completely loses batch information
        return self.classifer(x)


def get_dataloaders(batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    train_dataset, val_dataset = random_split(train_dataset, lengths=[0.8, 0.2])

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


def train(model, train_loader, val_loader, num_epochs, optimizer, scheduler, device):
    model.train()
    best_val_acc = float("-inf")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = F.cross_entropy(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch: {epoch + 1}: lr before step: {scheduler.get_last_lr()}")
        scheduler.step(val_acc)
        print(f"Epoch: {epoch + 1}: lr after step: {scheduler.get_last_lr()}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch:[{epoch+1}/{num_epochs}], Avg. Loss: {avg_loss:.2f}")


def evaluate(model, test_loader, device):
    model.eval()
    correct_preds = 0

    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data, targets = data.to(device), targets.to(device)
            predictions = model(data).max(1)[1]
            correct_preds += (predictions == targets).sum()

    accuracy = 100 * correct_preds / len(test_loader.dataset)
    return accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    model = CNNPabloVModuleList()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        batch_size=batch_size
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.1,
        patience=3,
    )
    train(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    accuracy = evaluate(
        model=model,
        test_loader=test_dataloader,
        device=device,
    )
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
