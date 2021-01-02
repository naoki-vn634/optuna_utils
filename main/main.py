import argparse
import json
import joblib
import os
import optuna
import sys
import torch
import random
import pickle
import torch.nn as nn

optuna.logging.disable_default_handler()
torch.manual_seed(0)

from distutils.util import strtobool
from sklearn.model_selection import train_test_split
from glob import glob

sys.path.append("../models/")
from model import CustomDensenet


sys.path.append("../learner/")
from trainer import SelfTrainer

sys.path.append("../utils/")
from utils import EarlyStopping, get_optimizer, get_dataloader


def train(
    net,
    dataloaders_dict,
    output,
    num_epoch,
    optimizer,
    scheduler,
    criterion,
    device,
    tblogger,
):
    FLAG = 0
    phase_list = ["train", "test"]
    Loss = {"train": [0] * num_epoch, "test": [0] * num_epoch}
    Acc = {"train": [0] * num_epoch, "test": [0] * num_epoch}

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epoch):
        print("Epoch:{}/{}".format(epoch + 1, num_epoch))
        print("-----------")

        for phase in phase_list:
            if (phase == "train") and (epoch == 0):
                continue
            else:
                print(phase)
                epoch_loss = 0
                epoch_correct = 0

                if phase == "train":
                    net.train()
                    torch.set_grad_enabled(True)
                else:
                    net.eval()
                    torch.set_grad_enabled(False)
                for images, labels in dataloaders_dict[phase]:

                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    out, middle = net(images)

                    loss = criterion(out, labels)
                    _, preds = torch.max(out, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += float(loss.item()) * images.size(0)
                    epoch_correct += torch.sum(preds == labels.data)

                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_correct.double() / len(
                    dataloaders_dict[phase].dataset
                )

                Loss[phase][epoch] = epoch_loss
                Acc[phase][epoch] = epoch_acc
                print("{} Loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

                if not tblogger is None:
                    tblogger.add_scalar("{}/Loss".format(phase), epoch_loss, epoch)

        scheduler.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)

    if args.tfboard:
        from torch.utils.tensorboard import SummaryWriter

        tf_dir = os.path.join(args.output, "tfboard/")
        if not os.path.isdir(tf_dir):
            os.makedirs(tf_dir)
        tblogger = SummaryWriter(tf_dir)
    else:
        tblogger = None

    # Load Dataset
    dataloaders_dict = get_dataloader(args.input, args.batchsize)

    # Model Definition

    net = CustomDensenet(num_classes=args.n_cls)
    net.to(device)

    for name, param in net.named_parameters():
        param.require_grad = True  # Finetuning

    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=0.1
    )
    criterion = nn.CrossEntropyLoss()

    train(
        net=net,
        dataloaders_dict=dataloaders_dict,
        output=args.output,
        num_epoch=args.epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        tblogger=tblogger,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/mnt/aoni02/matsunaga/dense/CONTINUAL"
    )
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpuid", type=str, default="0")
    parser.add_argument("--n_cls", type=int, default=3)
    parser.add_argument("--transfer", type=strtobool, default=False)
    parser.add_argument("--tfboard", type=strtobool, default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    main()
