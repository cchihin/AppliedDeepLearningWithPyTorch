import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
from datasets import load_from_disk


class ConvNet(nn.Module):

    def __init__(self, dropout_prob):
        super(ConvNet, self).__init__()
        # 1st Convolution layer
        # Input: (32, 32, 3)
        # Convolution layer: 10 filters, 3 widths, 1 stride and pad
        # Output: (32, 32, 10)
        # nn.Conv2d(in_channls, out_channels, width, stride, pad)    
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)

        # 1st Pooling layer
        # Input: (32, 32, 10)
        # Convolution layer: 2 widths, 2 strides 
        # Output: (16, 16, 10)
        # nn.MaxPool2d(width, stride)    
        self.pool1 = nn.MaxPool2d(2,2)

        # 2nd Convolution layer
        # Input: (16, 16, 10)
        # Convolution layer: 20 filters, 3 widths, 1 stride and pad
        # Output: (16, 16, 20)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)

        # 2nd Pooling layer
        # Input: (16, 16, 20)
        # Convolution layer: 2 widths, 2 strides 
        # Output: (8, 8, 20)
        self.pool2 = nn.MaxPool2d(2,2)

        # 3rd Convolution layer
        # Input: (8, 8, 20)
        # Convolution layer: 40 filters, 3 widths, 1 stride and pad
        # Output: (8, 8, 40)
        self.conv3 = nn.Conv2d(20, 40, 3, 1, 1)

        # 3rd Pooling layer
        # Input: (8, 8, 40)
        # Convolution layer: 2 widths, 2 strides 
        # Output: (4, 4, 40)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(4 * 4 * 40, 100)

        self.linear2 = nn.Linear(100, 10)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.pool3(F.relu(self.conv3(x)))

        # Flattening
        x = x.view(-1, 40 * 4 * 4) # dim0 is always batch size, so let pytorch infer
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        z = F.log_softmax(self.linear2(x), dim=1)

        return z

class HFCifar10Dataset(Dataset):
    """Wraps a Hugging Face CIFAR10 split (PIL 'img' + 'label') as a torch Dataset."""

    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        image = example['img']
        if self.transform:
            image = self.transform(image)
        return image, example['label']


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    # Loading dataset downloaded from Hugging Face (uoft-cs/cifar10)
    batch_size = 100

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    prob_flip = 0.5
    prob_gray = 0.1 

    transform_list = [
            {
            "train": transforms. Compose([
                transforms.RandomHorizontalFlip(prob_flip),
                transforms.RandomGrayscale(prob_gray),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

            "test": transforms. Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            },
            {
            "train": transforms. Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

            "test": transforms. Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            }
        ]

    fig, ax = plt.subplots(len(transform_list),2)

    k = 0
    for transform in transform_list:

        hf_dataset = load_from_disk('../Activity7/data/cifar10_hf')

        print('Loading CIFAR10 train_data')
        train_data = HFCifar10Dataset(hf_dataset['train'], transform=transform["train"])

        print('Loading CIFAR10 test_data')
        test_data = HFCifar10Dataset(hf_dataset['test'], transform=transform["test"])

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
                                  num_workers = 16)

        valid_loader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   sampler=valid_sampler,
                                  num_workers = 16)

        test_loader = DataLoader(test_data,
                                  batch_size=batch_size)


        # Initialising networks
        dropout = 0.2
        model = ConvNet(dropout)
        # Pushing to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        # Running training cycle
        epochs = 100
        train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

        for e in range(epochs):

            start_time = time.time()
            running_loss = 0
            running_acc  = 0
            itx = 0

            model.train()
            for batch, target in train_loader:

                batch, target = batch.to(device), target.to(device)
                batch_start = time.time()
                y_pred = model(batch)
                criteria = criterion(y_pred, target)
                optimiser.zero_grad()
                criteria.backward()
                optimiser.step()

                running_loss += criteria.item()

                prob = torch.exp(y_pred)
                top_p, top_class = prob.topk(1, dim=1)
                running_acc += accuracy_score(target.cpu(), top_class.cpu().squeeze())

                itx += 1
                # print(f'Training iteration: {itx}, that took: {time.time()-batch_start}')
            
            train_losses.append(running_loss/itx)
            train_accs.append(running_acc/itx * 100)

            running_loss = 0
            running_acc  = 0
            itx = 0

            model.eval()
            with torch.no_grad():
                
                for batch, target in valid_loader:
                    batch, target = batch.to(device), target.to(device)
                    y_pred_valid = model(batch)
                    criteria = criterion(y_pred_valid, target)

                    running_loss += criteria.item()

                    prob = torch.exp(y_pred_valid)
                    top_p, top_class = prob.topk(1, dim=1)
                    running_acc += accuracy_score(target.cpu(), top_class.cpu().squeeze())
                    
                    itx += 1

            valid_losses.append(running_loss/itx)
            valid_accs.append(running_acc/itx * 100)

            checkpoint = {"dropout": dropout, "state_dict": model.state_dict()}
            torch.save(checkpoint, f"tf_dropout{dropout}_batch{batch_size}.pth")

            print(f'Epoch: {e}, time taken: {time.time()-start_time}')
            print(f'Training loss: {train_losses[-1]:.3f}')
            print(f'Training accuracy: {train_accs[-1]:.3f}')
            print(f'Validation loss: {valid_losses[-1]:.3f}')
            print(f'Validation accuracy: {valid_accs[-1]:.3f}')
            print(f'')

        # Evaluting test accuracies and losses
        test_loss = 0
        test_acc = 0
        itx = 0
        model.eval()
        with torch.no_grad():
            for batch, target in test_loader:
                batch, target = batch.to(device), target.to(device)
                y_test = model(batch)
                criteria = criterion(y_test, target)

                test_loss += criteria.item()

                prob = torch.exp(y_test)
                top_p, top_class = prob.topk(1, dim=1)
                test_acc += accuracy_score(target.cpu(), top_class.cpu().squeeze())

                itx += 1

            test_loss = test_loss / itx
            test_acc = test_acc / itx * 100

        ax[k][0].plot(range(epochs), train_losses, '-o', label='Train')
        ax[k][0].plot(range(epochs), valid_losses, '-o', label='Valid')
        ax[k][0].set_xlabel('Epochs')
        ax[k][0].set_ylabel('Losses')
        ax[k][0].grid()
        if k == 0:
            ax[k][0].set_title('With augmentation' + '\n' + f'Test Loss: {test_loss:.3f}')
        else:
            ax[k][0].set_title('Without augmentation' + '\n' + f'Test Loss: {test_loss:.3f}')

        ax[k][1].plot(range(epochs), train_accs, '-o', label='Train')
        ax[k][1].plot(range(epochs), valid_accs, '-o', label='Valid')
        ax[k][1].legend()
        ax[k][1].set_xlabel('Epochs')
        ax[k][1].set_ylabel('Accuracy (%)')
        ax[k][1].grid()
        if k == 0:
            ax[k][1].set_title('With augmentation' + '\n' + f'Test Acc: {test_acc:.3f}')
        else:
            ax[k][1].set_title('Without augmentation' + '\n' + f'Test Acc: {test_acc:.3f}')

        k += 1

    # fig.set_size_inches(10*len(transform_list),6)
    fig.set_size_inches(10,6*len(transform_list))
    fig.savefig(f'tf_dropout{dropout}_batch{batch_size}_compare.pdf', bbox_inches='tight')
