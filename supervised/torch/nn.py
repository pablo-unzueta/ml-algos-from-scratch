import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 50):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def data_loaders(batch_size):
    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.reshape(data.shape[0], -1).to(device)
            targets = targets.to(device)

            # forward pass
            scores = model(data)
            loss = F.cross_entropy(scores, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    num_correct, num_samples = 0, 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            # flatten
            data = data.reshape(data.shape[0], -1)

            # forward pass
            scores = model(data)
            _, predictions = scores.max(1)  # get max_values, indices along dimenion 1
            num_correct += (predictions == targets).sum()
            num_samples += predictions.shape[0]

    accuracy = num_correct / num_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100

    train_loader, test_loader = data_loaders(batch_size)

    model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )
    evaluate(model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
