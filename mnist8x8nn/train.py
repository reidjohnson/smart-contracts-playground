import argparse
import os
import pickle
import re
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import mnist

# Fixes: OMP Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


def do_train(model, n_epochs=100, batch_size=256):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for idx, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X_train.float())
            loss = loss_fn(y_pred, y_train.long())
            loss.backward()
            optimizer.step()

        n_correct = 0
        n_total = 0
        model.eval()
        for idx, (X_test, y_test) in enumerate(test_loader):
            y_pred = model(X_test.float()).detach()
            y_pred = np.argmax(y_pred, axis=-1)
            label_np = y_test.numpy()
            y_acc = y_pred == y_test
            n_correct += np.sum(y_acc.numpy(), axis=-1)
            n_total += y_acc.shape[0]

        print("Epoch: {:3d}, loss: {:.5f}, acc: {:.4f}".format(
            epoch, loss.sum().item(), n_correct / n_total))

    return model


def format_ndarray_for_solidity(a, dtype="int128"):
    a = str(repr(a))
    a = re.sub(r'^array\(', r'', a)
    a = re.sub(r'^\s*$', r'', a, flags=re.MULTILINE)
    a = re.sub(r'^\s*', r'\t', a, flags=re.MULTILINE)
    a = re.sub(r'^(\s*)(\[{1,})\[', r'\2\n\1[', a, flags=re.MULTILINE)
    a = re.sub(r'(\]{1,})\]', r']\n\1', a)
    a = re.sub(r'\], dtype=.*\)', r']', a)
    a = re.sub(r'\[([-,\s]*\d*),', r'[{}(\1),'.format(dtype), a)
    a = a + ';'
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the model.")
    parser.add_argument("--write-arrays", action="store_false",
                        help="Write arrays to text files.")
    args = parser.parse_args()

    train_dataset = mnist.MNIST(
        root="./train",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((8, 8)),
            transforms.ToTensor()
        ])
    )
    test_dataset = mnist.MNIST(
        root="./test",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((8, 8)),
            transforms.ToTensor()
        ])
    )

    model = Model()

    if args.retrain:
        model = do_train(model, n_epochs=10, batch_size=256)
        with open("model.pkl", 'wb') as f:
            pickle.dump(model.state_dict(), f)

    with open("model.pkl", 'rb') as f:
        weights = pickle.load(f)
    for param, weight in zip(model.parameters(), weights.values()):
        param.data = weight

    if args.write_arrays:
        with open("sol_weights.txt", 'w') as f:
            for w_name, w in weights.items():
                w = (w.numpy() * 255).astype(np.int32)
                w = format_ndarray_for_solidity(w)
                f.write(w_name + "\n")
                f.write(w + "\n")
                f.write("\n")

    if args.write_arrays:
        test_loader = DataLoader(test_dataset, batch_size=1)
        _, (X_digit, y_digit) = next(enumerate(test_loader))

        X_digit = (X_digit.numpy().reshape((8, 8)) * 255).astype(np.uint8)

        with open("sol_ex_digit.txt", 'w') as f:
                digit = format_ndarray_for_solidity(X_digit)
                f.write(digit)
                f.write("\n")

        #Image.fromarray(X_digit).convert('RGB').resize((32, 32)).show()
