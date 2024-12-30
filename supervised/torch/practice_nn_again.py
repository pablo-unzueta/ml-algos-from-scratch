import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_dataloaders(batch_size: int):
    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=False
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class NeuralNetwork(nn.Module):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 50, num_classes: int = 10
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, num_epochs, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, targets in tqdm(train_loader):
            data = data.reshape(data.shape[0], -1).to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = F.cross_entropy(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_loss:.2f}")


def evaluate(model, test_loader, device):
    model.eval()
    num_correct = 0
    for data, targets in tqdm(test_loader):
        data = data.reshape(data.shape[0], -1).to(device)
        targets = targets.to(device)

        predictions = model(data).max(1)[1]
        num_correct += (predictions == targets).sum()

    accuracy = 100 * num_correct / len(test_loader.dataset)
    print(f"Accuracy: {accuracy:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 3

    model = NeuralNetwork()
    model = model.to(device)
    train_loader, test_loader = get_dataloaders(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        device=device,
    )
    evaluate(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
