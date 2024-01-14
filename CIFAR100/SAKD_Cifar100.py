from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import distiller
import load_settings


gpu_num = 0

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--save_path', type=str, default='./data/xxx.pth')
parser.add_argument('--paper_setting', default='a', type=str, help='a-j')
parser.add_argument('--device', default=gpu_num, type=int, help='use gpu')
parser.add_argument('--epochs', default=240, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--temperature', default=4, type=float, help='temperature')
parser.add_argument('--alpha', default=0.9, type=float, help='weight for KD (Hinton)')
parser.add_argument('--beta', default=200, type=float)
args = parser.parse_args()


time_all = 0
max_acc = 0

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0),
])


trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)



# Model
t_net, s_net, args = load_settings.load_paper_settings(args)

# Module for distillation
d_net = distiller.Distiller(t_net, s_net)

print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    torch.cuda.set_device(gpu_num)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True


criterion_CE = nn.CrossEntropyLoss()
criterion_kl = distiller.DistillKL(args.temperature)


# Training
def train_with_distill(d_net, epoch):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global time_all
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        s_outputs, t_outputs, loss_distill = d_net(inputs)
        loss_KL = criterion_kl(s_outputs, t_outputs)
        loss_KD = distiller.KD(s_outputs, targets)
        loss_SAD = loss_distill.sum()

        loss = loss_KD + args.beta * loss_SAD + args.alpha * loss_KL

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(s_outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx
    time_all += (time.time() - epoch_start_time)
    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)


def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct


print('Performance of teacher network ')
test(t_net)

for epoch in range(args.epochs):

    if epoch == 0:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == 150:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == 180:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr * 0.01, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == 210:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr * 0.001, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_loss = train_with_distill(d_net, epoch)
    test_loss, accuracy = test(s_net)
    if accuracy >= max_acc:
        max_acc = accuracy


print('Train \t Time Taken: %.2f sec /epoch' % (time_all/args.epochs))
print('Performance of student network')
print(max_acc)