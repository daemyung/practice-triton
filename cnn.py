# MIT License
#
# Copyright (c) 2023 Daemyung Jang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.hub as hub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        factory_kwargs = {"device": device}
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, **factory_kwargs),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, **factory_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128, **factory_kwargs),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10, **factory_kwargs),
            nn.Softmax(1),
        )

    def forward(self, input):
        return self.model(input)


def train(epoch, num_epochs, dataloader, model, criterion, optimizer, device):
    model.train()
    progress = hub.tqdm(dataloader, desc=f"[TRAIN] {epoch+1}/{num_epochs}")
    total_loss = 0

    for idx, (inputs, labels) in enumerate(progress):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        progress.set_postfix({"Loss": f"{loss:.4f}"})
        total_loss += loss

    print(f"[TRAIN] {epoch+1}/{num_epochs} Average Loss: {total_loss/len(dataloader):.4f}")


def test(epoch, num_epochs, dataloader, model, device):
    model.eval()
    progress = hub.tqdm(dataloader, desc=f"[TEST] {epoch+1}/{num_epochs}")
    total_correct = 0
    num_inputs = 0

    for idx, (inputs, labels) in enumerate(progress):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = torch.argmax(outputs, 1)
        correct = torch.sum(torch.eq(outputs, labels))
        progress.set_postfix({"Accuracy": f"{100 * correct / labels.numel():.2f}"})
        total_correct += correct
        num_inputs += labels.numel()

    print(f"[TEST] {epoch+1}/{num_epochs} Average Accuracy: {total_correct / num_inputs * 100:.2f}")


def main():
    num_epochs = 4
    num_batches = 256

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    train_dataset = datasets.MNIST("./datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST("./datasets", train=False, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, num_batches, True)
    test_dataloader = DataLoader(test_dataset, num_batches, True)

    cnn = CNN(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())

    for epoch in range(0, num_epochs):
        train(epoch, num_epochs, train_dataloader, cnn, criterion, optimizer, device)
        test(epoch, num_epochs, test_dataloader, cnn, device)

    inputs, labels = next(iter(train_dataloader))
    num_rows = 3
    num_cols = 5
    num_inputs = num_rows * num_cols
    inputs = inputs[:num_inputs,]
    labels = labels[:num_inputs,]
    outputs = cnn(inputs.to(device))
    outputs = torch.argmax(outputs, 1)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5, 5))
    axs = axs.ravel()

    for i in range(num_inputs):
        axs[i].imshow(torch.permute(inputs[i], (1, 2, 0)).numpy())
        axs[i].set_title(f"{labels[i]}/{outputs[i]}")
        axs[i].axis("off")

    plt.show()


if __name__ == "__main__":
    main()
