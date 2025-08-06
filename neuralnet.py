# Code in this file largely taken from:
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021 (from Membership Inference Attacks From First Principles)
# and https://opacus.ai/tutorials/building_image_classifier

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import CIFAR10

import opacus
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
DATA_ROOT = '../cifar10'

def get_CIFAR10():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader

def accuracy(preds, labels):
    return (preds == labels).mean()

def train_epoch(model, train_loader, optimizer, epoch, privacy_engine):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    losses = []
    top1_acc = []

    
    if privacy_engine is not None:
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (images, target) in tqdm(enumerate(memory_safe_data_loader)):   
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

                loss.backward()
                optimizer.step()

                if (i+1) % 200 == 0:
                    epsilon = privacy_engine.get_epsilon() if privacy_engine is not None else 0.0
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f})"
                    )
    else:
        memory_safe_data_loader = train_loader
        for i, (images, target) in tqdm(enumerate(memory_safe_data_loader)):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon() if privacy_engine is not None else 0.0
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f})"
                )

def train_model(train_loader, epochs=10, eps=1.0, private=True):
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    if private:
        privacy_engine = opacus.PrivacyEngine()

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=eps,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )

    for epoch in tqdm(range(epochs)):
        train_epoch(model, train_loader, optimizer, epoch + 1, privacy_engine if private else None)

    return model

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

if __name__ == "__main__":
    train_data, test_data = get_CIFAR10()
    # nonprivate_model = train_model(train_data, epochs=5, private=False)
    # print(f"Non-private test accuracy: {test(nonprivate_model, test_data)}")
    private_model = train_model(train_data, epochs=5, eps=1.0)
    print(f"Private test accuracy: {test(private_model, test_data)}")
    