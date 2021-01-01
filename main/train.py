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


def objective(trial):
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
    hidden_size = int(trial.suggest_discrete_uniform("hidden_size", 128, 512, 128))

    net = CustomDensenet(num_classes=args.n_cls, hidden_size=hidden_size)
    net.to(device)

    for name, param in net.named_parameters():
        param.require_grad = True  # Finetuning

    optimizer = get_optimizer(trial, net)
    criterion = nn.CrossEntropyLoss()

    flooding_level = float(
        trial.suggest_discrete_uniform("flooding_level", 0.00, 0.20, 0.02)
    )

    trainer = SelfTrainer(
        model=net,
        dataloaders_dict=dataloaders_dict,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output=args.output,
        tblogger=tblogger,
        flooding=flooding_level,
    )

    ES = EarlyStopping(patience=15, verbose=1)

    best = 100
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        for phase in ["train", "test"]:
            if (epoch == 0) and (phase == "train"):
                continue
            if phase == "train":
                loss, acc = trainer.train(phase, epoch)
            elif phase == "test":
                loss, acc, error_rate = trainer.eval(phase, epoch)

        if error_rate < best:
            best = error_rate
            best_ep = epoch + 1

        if ES.validate(loss):
            print("end loop")
            break

    return best


def main(args):
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.trial_size)

    # with open(os.path.join(args.output, "best_param.json"), "w") as f:
    #     json.dump(study.best_params, f, indent=4)

    joblib.dump(study, os.path.join(args.output, "study.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/mnt/aoni02/matsunaga/dense/CONTINUAL"
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--trial_size", type=int, default=100)
    parser.add_argument("--output", type=str)
    parser.add_argument("--gpuid", type=str, default="0")
    parser.add_argument("--n_cls", type=int, default=3)
    parser.add_argument("--transfer", type=strtobool, default=False)
    parser.add_argument("--tfboard", type=strtobool)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "params.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)
