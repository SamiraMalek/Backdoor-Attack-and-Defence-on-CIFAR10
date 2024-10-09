from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

from resnet import ResNet18
#from utils import pattern_craft, add_backdoor

#parser = argparse.ArgumentParser(description='PyTorch Backdoor Attack Crafting')
#args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Load attack configuration
#with open('/content/config.json') as config_file:
#    config = json.load(config_file)
config = {
  "SC": 8,
  "TC": 4,
  "BD_NUM": 300,
  "PATTERN_TYPE": "pixel",
  "PERTURBATION_SIZE": 0.1
}

def pattern_craft(im_size, pattern_type, perturbation_size):
    N = 4
    perturbation = torch.zeros(im_size) 
    if pattern_type == "pixel":
      for i in range(0,N):
        random_row = random.randint(3, im_size[1]-3)
        random_column = random.randint(3, im_size[2]-3)
        random_RGB = random.randint(0, im_size[0]-1)
        random_factor = random.gauss(1, 0.05)
        print("i=",i, random_row, random_column, random_RGB, random_factor)
        perturbation[random_RGB, random_row, random_column] += perturbation_size * random_factor #perturbation_size 
        #perturbation += 0.1
        perturbation = torch.clamp(perturbation, min=0, max=1)
    return perturbation

def add_backdoor(image, perturbation):

    image_perturbed = image + perturbation
    image_perturbed = torch.clamp(image_perturbed, min=0, max=1)

    return image_perturbed

# Load raw data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='/scratch/sxm6547/Advrl/project3/l2one/data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='/scratch/sxm6547/Advrl/project3/l2one/data', train=False, download=True, transform=transform_test)

# Create the backdoor pattern (TO DO)
perturbation = pattern_craft(trainset.__getitem__(0)[0].size(), config['PATTERN_TYPE'], config['PERTURBATION_SIZE'] )
#img_to_plot = perturbation.permute(1, 2, 0)
#plt.imshow(img_to_plot)
#plt.title("Image with shape [3, 32, 32]")
#plt.axis('off')  # Hide axes ticks
#plt.show()

# Crafting training backdoor images
train_images_attacks = None
train_labels_attacks = None
ind_train = [i for i, label in enumerate(trainset.targets) if label==config["SC"]]
ind_train = np.random.choice(ind_train, config["BD_NUM"], False)
for i in ind_train:
    if train_images_attacks is not None:
        train_images_attacks = torch.cat([train_images_attacks, add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0) #(TO DO)
        train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([config["TC"]], dtype=torch.long)], dim=0)
    else:
        train_images_attacks = add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)
        train_labels_attacks = torch.tensor([config["TC"]], dtype=torch.long)

# Crafting test backdoor images
test_images_attacks = None
test_labels_attacks = None
ind_test = [i for i, label in enumerate(testset.targets) if label==config["SC"]]
for i in ind_test:
    if test_images_attacks is not None:
        test_images_attacks = torch.cat([test_images_attacks, add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0)
        test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([config["TC"]], dtype=torch.long)], dim=0)
    else:
        test_images_attacks = add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)
        test_labels_attacks = torch.tensor([config["TC"]], dtype=torch.long)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='/scratch/sxm6547/Advrl/project3/l2one/data', train=True, download=True, transform=transform_train)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load in attack data
'''
if not os.path.isdir('attacks'):
    print ('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./attacks/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
test_attacks = torch.load('./attacks/test_attacks')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']
'''
# Normalize backdoor test images
testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)

# Poison the training set and remove clean images used for creating backdoor training images
## TO DO ##

all_indices = set(range(len(trainset)))
used_indices = set(ind_train)
remaining_indices = list(all_indices - used_indices)

# Create a subset of the original trainset excluding the backdoored images
remaining_trainset = torch.utils.data.Subset(trainset, remaining_indices)

# Step 3: Combine the original reduced dataset with the new dataset of backdoored images
# Convert backdoored images and labels into a TensorDataset
backdoor_dataset = torch.utils.data.TensorDataset(train_images_attacks, train_labels_attacks)

# Concatenate the two datasets
# Since PyTorch doesn't have a direct concatenate method for datasets, use ConcatDataset
combined_dataset = torch.utils.data.ConcatDataset([remaining_trainset, backdoor_dataset])


# Load in the datasets
def custom_collate_fn(batch):
    # Split data and labels
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Convert data and labels to tensors if they are not already
    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return data, labels

trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
#trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=100, shuffle=False, num_workers=1)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

resume = False
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_contam'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_contam/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 65:
        lr *= 1e-3
    elif epoch > 55:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)

    return lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_scheduler(epoch))
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

    acc = 100. * correct / total
    print("train_loss=", train_loss, "train_loss/total=", train_loss/total,"train_loss/batch_idx=",train_loss/batch_idx)
    print('Train ACC: %.3f' % acc)

    return net


# Test
def test(epoch):
    global best_acc
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

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/scratch/sxm6547/Advrl/project3/l2one/checkpoint_contam'):
            os.mkdir('/scratch/sxm6547/Advrl/project3/l2one/checkpoint_contam')
        torch.save(state, '/scratch/sxm6547/Advrl/project3/l2one/checkpoint_contam/ckpt.pth')
        best_acc = acc


# Test ASR
def test_attack(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Attack success rate: %.3f' % acc)


for epoch in range(start_epoch, start_epoch+70):
    model_contam = train(epoch)
    test(epoch)
    test_attack(epoch)

    # Save model
    if not os.path.isdir('/scratch/sxm6547/Advrl/project3/l2one/contam'):
        os.mkdir('/scratch/sxm6547/Advrl/project3/l2one/contam')
    torch.save(model_contam.state_dict(), '/scratch/sxm6547/Advrl/project3/l2one/contam/model_contam.pth')