from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import wandb
import optuna

from plr_exercise.modules.cnn import Net


def train(args, model, device, train_loader, optimizer):
    """Function for training 

    Argments:
        args: experiment args
        model: model needed to train
        device: 'cuda' or 'cpu'
        train_loader: dataloader of the training set
        optimizer: optimizer of training

    Return:
        train_loss: avg loss of training
        train_acc: accuracy of prediction in range [0, 1]
    """
    model.train()

    train_loss = 0
    train_acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # update loss and accuracy
        train_loss += loss.item() / len(train_loader.dataset)
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item() / len(
            train_loader.dataset
        )

        if batch_idx % args.log_interval == 0:
            print(
                "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break

    return train_acc, train_loss


def test(model, device, test_loader):
    """Function for testing 

    Argments:
        model: model to be tested
        device: 'cuda' or 'cpu'
        train_loader: dataloader of the testing set

    Return:
        test_loss: avg loss of testing
        test_acc: accuracy of prediction in range [0, 1]
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * test_acc,
        )
    )

    return test_acc, test_loss


def get_mnist(args, use_cuda):
    """Obtain the MNIST dataset and create dataloader

    Argments:
        args: experiment args
        use_cuda: if use cuda
        
    Return:
        train_loader: dataloader of the training set
        test_loader: dataloader of the testing set
    """

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # data preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # download dataset
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    # create dataloader
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def optuna_objective(trial, args, use_cuda):
    """Objective function for Optuna study

    Argments:
        trial: built-in object of optuna study
        args: experiment args
        use_cuda: if use cuda
        
    Return:
        test_acc: accuracy of testing, to be maximized
    """
    # get shuffled dataset
    train_loader, test_loader = get_mnist(args, use_cuda)

    # define model
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model = Net().to(device)

    # define optuna parameters
    epochs = trial.suggest_int("epochs", 5, 15)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train
    for epoch in range(epochs):
        # print(f"====== Epoch: [{epoch}/{epochs}] ======\n")
        train_acc, train_loss = train(args, model, device, train_loader, optimizer)

        trial.report(train_acc, epoch)
        scheduler.step()
    # test
    test_acc, test_loss = test(model, device, test_loader)

    return test_acc


def main():
    """
    Main function: train a network for MNIST dataset
    """
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # ============ finding the best hyperparameter ============
    # searching for lr and epochs that maximize the train_acc
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner
    )
    study.optimize(
        lambda trial: optuna_objective(trial, args, use_cuda),
        n_trials=5,
    )

    optuna_lr = study.best_params["lr"]
    optuna_epochs = study.best_params["epochs"]

    print(f"Best learning rate found by optuna: {optuna_lr}")
    print(f"Best epochs found by optuna: {optuna_epochs}")

    # start a new wandb run to track this script
    wandb.init(
        project="plr-exercise",
        name="MNIST_Run_Optuna",
        # track hyperparameters and run metadata
        config={
            "dataset": "CIFAR-100",
            "architecture": "CNN",
            "optimizer": "Adam",
            "learning_rate": optuna_lr,
            "scheduler": "StepLR",
            "gamma": args.gamma,
            "epochs": optuna_epochs,
        },
    )

    # prepare dataset
    train_loader, test_loader = get_mnist(args, use_cuda)

    # define model
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model = Net().to(device)
    # define optimizer and schedule
    optimizer = optim.Adam(model.parameters(), lr=optuna_lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train & test
    for epoch in range(optuna_epochs):
        print(f"====== Epoch: [{epoch}/{optuna_epochs}] ======\n")
        # train
        train_acc, train_loss = train(args, model, device, train_loader, optimizer)
        wandb.log(
            {
                "Train_epoch": epoch,
                "train_acc": train_acc * 100.0,
                "train_loss": train_loss,
            }
        )
        # test
        test_acc, test_loss = test(model, device, test_loader)
        wandb.log(
            {"Test_epoch": epoch, "test_acc": test_acc * 100.0, "test_loss": test_loss}
        )
        # change lr
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")

    # add artifacts
    code_artifact = wandb.Artifact("training_code", type="code")
    code_artifact.add_file("scripts/train.py")
    wandb.log_artifact(code_artifact)
    # finish
    wandb.finish()


if __name__ == "__main__":
    main()
