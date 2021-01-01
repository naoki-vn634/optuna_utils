import json
import os
import sys
import torch
import torch.optim as optim

sys.path.append("../preprocess/")
from preprocess import ImageTransform, PatientDataset


class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float("inf")
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def get_optimizer(trial, model):
    optimizer_names = ["Adam", "SGD"]
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_names)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)

    if optimizer_name == optimizer_names[0]:
        adam_lr = trial.suggest_loguniform("adam_lr", 1e-4, 1e-3)
        optimizer = optim.Adam(
            model.parameters(), lr=adam_lr, weight_decay=weight_decay
        )

    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform("momentum_sgd_lr", 1e-4, 1e-3)
        optimizer = optim.SGD(
            model.parameters(),
            lr=momentum_sgd_lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    return optimizer


def get_dataloader(input, batchsize):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = ImageTransform(mean, std)

    # Load Data from JSON FILE
    with open(os.path.join(input, "database.json"), "r") as f:
        database = json.load(f)
    x_train = database["train"]["path"]
    x_test = database["test"]["path"]
    y_train = database["train"]["label"]
    y_test = database["test"]["label"]

    print("## Label")
    print("## Train")
    print("|-- Yes: ", y_train.count(1))
    print("|-- No : ", y_train.count(0))
    print("|-- Garbage: ", y_train.count(2))
    print("## Test")
    print("|-- Yes: ", y_test.count(1))
    print("|-- No : ", y_test.count(0))
    print("|-- Garbage: ", y_test.count(2))

    train_dataset = PatientDataset(
        x_train, y_train, transform=transforms, phase="train", color=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, num_workers=1, shuffle=True
    )

    test_dataset = PatientDataset(
        x_test, y_test, transform=transforms, phase="test", color=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, num_workers=1, shuffle=False
    )

    print("Train_Length: ", len(train_dataloader.dataset))
    print("Test_Length: ", len(test_dataloader.dataset))

    dataloaders_dict = {"train": train_dataloader, "test": test_dataloader}
    return dataloaders_dict
