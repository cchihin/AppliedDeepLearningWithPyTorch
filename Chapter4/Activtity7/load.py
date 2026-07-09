import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
from datasets import load_from_disk
from train import ConvNet, HFCifar10Dataset
checkpoint = torch.load("checkpoint.pth")

model = ConvNet()

model.load_state_dict(checkpoint['state_dict'])

torch.manual_seed(0)
np.random.seed(0)

# Loading dataset downloaded from Hugging Face (uoft-cs/cifar10)
batch_size = 10000
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

hf_dataset = load_from_disk('data/cifar10_hf')

print('Loading CIFAR10 train_data')
train_data = HFCifar10Dataset(hf_dataset['train'], transform=transform)

print('Loading CIFAR10 test_data')
test_data = HFCifar10Dataset(hf_dataset['test'], transform=transform)

valid_size = 0.2
# Shuffling indexes
idx = list(range(len(train_data)))
np.random.shuffle(idx)
split_size = int(np.floor(valid_size * len(train_data)))
train_idx, valid_idx = idx[split_size:], idx[:split_size]

# Sampling training dataset
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_loader = DataLoader(train_data,
                           batch_size=batch_size,
                           sampler=train_sampler,
                          num_workers = 2)

valid_loader = DataLoader(train_data,
                           batch_size=batch_size,
                           sampler=valid_sampler,
                          num_workers = 2)

test_loader = DataLoader(test_data,
                          batch_size=batch_size)

criterion = nn.NLLLoss()

with torch.no_grad():
    
    running_loss, running_acc, itx = 0, 0, 0
    for batch, target in test_loader:
        y_pred_valid = model(batch)
        criteria = criterion(y_pred_valid, target)

        running_loss += criteria.item()

        prob = torch.exp(y_pred_valid)
        top_p, top_class = prob.topk(1, dim=1)
        running_acc += accuracy_score(target, top_class)
        
        itx += 1

import matplotlib.pyplot as plt
from PIL import Image

fig, axex = plt.subplots(3,3)
axes = axex.flatten()

itx = 0
for ax in axes:

    ax.imshow(
