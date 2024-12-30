import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def get_dataloaders(batch_size):
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


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int = 10):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, num_epochs, optimizer, device):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data, targets = (
                data.reshape(data.shape[0], -1).to(device),
                targets.to(device),
            )

            # forward pass
            scores = model(data)
            loss = F.cross_entropy(scores, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.2f}")


def evaluate(model, test_loader, device):
    model.eval()
    correct_preds = 0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data, targets = (
                data.reshape(data.shape[0], -1).to(device),
                targets.to(device),
            )
            scores = model(data)
            _, predictions = scores.max(
                1
            )  # get max (value,index) , indices along dimenion 1
            predictions = scores.max(1)[1]

            correct_preds += (targets == predictions).sum()

    print(f"Accuracy: {100 * correct_preds/total_samples:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    input_size = 784
    hidden_size = 50
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 3

    model = NeuralNetwork(
        input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    )
    model = model.to(device)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=num_epochs,
        device=device,
    )

    evaluate(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
