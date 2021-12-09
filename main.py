'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--kc', default=64, type=int, help='model size')
parser.add_argument('--epoch', default=400, type=int, help='total training epochs')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--noise', default=0, type=int, help='label noise %')
parser.add_argument('--eval', action='store_true', help='only do evaluation')
parser.add_argument('--quiet', '-q', action='store_true', help='be quiet')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if not args.quiet: print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

do_download = not os.path.exists('./data')

if args.resume:
    # Load checkpoint.
    if not args.quiet: print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    args.kc = checkpoint['kc']
    args.noise = checkpoint['noise']
    args.epoch = checkpoint['end_epoch']

# Training data with optional noise
def flip_random_label(x):
    image, label = x
    wrong = list(range(10))
    del wrong[label]
    label = np.random.choice(wrong)
    x = image, label

    return x

noise_indices = []
noise_labels = []
if not args.eval:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=do_download, transform=transform_train)

    if args.noise != 0:
        # If resuming we want the label flips to be the same
        if args.resume:
            noise_indices = checkpoint['noise_indices']
            noise_labels = checkpoint['noise_labels']
        else:
            noise_frac = args.noise / 100
            num_noise_samples = int(noise_frac * len(trainset))
            if not args.quiet: print(f'Flipping {args.noise}% of labels ({num_noise_samples} samples)')
            noise_indices = np.random.choice(np.arange(len(trainset)), size=num_noise_samples, replace=False)
        noisy_data = [x for x in trainset]
        if args.resume:
            for label,index in zip(noise_labels, noise_indices):
                noisy_data[index] = (noisy_data[index][0], label)
        else:
            for i in noise_indices:
                noisy_data[i] = flip_random_label(noisy_data[i])
            noise_labels = [noisy_data[i][1] for i in noise_indices]
        trainset = noisy_data

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=do_download, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

# Model
if not args.quiet: print('==> Building model..')
net = PreActResNet18(args.kc)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch'] + 1

criterion = nn.CrossEntropyLoss()
# Adam with LR=0.0001
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if not args.quiet:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if not args.quiet:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    # Save checkpoint.
    if epoch % 10 == 0 and not args.eval:
        if not args.quiet: print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'kc': args.kc,
            'noise': args.noise,
            'noise_indices': noise_indices,
            'noise_labels': noise_labels,
            'end_epoch': args.epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/noise{args.noise}_kc{args.kc}_epoch{epoch}_ckpt.pth')
    return acc

if args.eval:
    if not args.resume:
        parser.error("--eval requires --resume CHECKPOINT")
    print(args.kc, args.noise, test(0))
else:
    for epoch in range(start_epoch, args.epoch+1):
        train(epoch)
        test(epoch)
