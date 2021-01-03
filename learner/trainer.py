import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SelfTrainer(nn.Module):
    def __init__(
        self,
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        device,
        output,
        tblogger,
        flooding,
    ):
        super(SelfTrainer, self).__init__()
        self.model = model
        self.dataloaders_dict = dataloaders_dict
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output = output
        self.tblogger = tblogger
        self.flooding = flooding

    def train(self, phase, epoch):
        torch.backends.cudnn.benchmark = True
        self.model.train()
        torch.set_grad_enabled(True)
        epoch_loss = 0
        epoch_correct = 0
        for images, labels in self.dataloaders_dict[phase]:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            output, middle = self.model(images)
            _, preds = torch.max(output, 1)

            loss = self.criterion(output, labels)
            if self.flooding > 0:
                loss = (loss - self.flooding).abs() + self.flooding
            loss.backward()

            epoch_loss += float(loss.item()) * images.size(0)
            epoch_correct += torch.sum(preds == labels.data)
            self.optimizer.step()

        epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
        epoch_acc = epoch_correct.double() / len(self.dataloaders_dict[phase].dataset)
        print("{} Loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

        if self.tblogger is not None:
            self.tblogger.add_scalar("{}/Loss".format(phase), epoch_loss, epoch)
            self.tblogger.add_scalar("{}/Acc".format(phase), epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def eval(self, phase, epoch):
        torch.backends.cudnn.benchmark = True
        self.model.eval()
        torch.set_grad_enabled(False)
        epoch_loss = 0
        epoch_correct = 0
        for images, labels in self.dataloaders_dict[phase]:
            images, labels = images.to(self.device), labels.to(self.device)

            output, middle = self.model(images)
            _, preds = torch.max(output, 1)

            loss = self.criterion(output, labels)

            epoch_loss += float(loss.item()) * images.size(0)
            epoch_correct += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
        epoch_acc = epoch_correct.double() / len(self.dataloaders_dict[phase].dataset)

        print("{} Loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

        if self.tblogger is not None:
            self.tblogger.add_scalar("{}/Loss".format(phase), epoch_loss, epoch)
            self.tblogger.add_scalar("{}/Acc".format(phase), epoch_acc, epoch)

        return epoch_loss, epoch_acc, 1 - epoch_acc
