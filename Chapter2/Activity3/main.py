import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

def dataloader():

    # 1. Load csv file
    # data = pd.read_csv('../Activity2/YearPredictionMSD.txt', header=None, nrows=50000)
    data = pd.read_csv('../Activity2/YearPredictionMSD.txt', header=None)

    # 2. Check for missing values 
    missing = data.isnull().sum().sum()
    print(f'Missing entries: {missing}')

    # 3. Checking for outliers
    outliers = {}
    for col in range(data.shape[1]):

        min_t = data[data.columns[col]].mean() - (3 * data[data.columns[col]].std() )
        max_t = data[data.columns[col]].mean() + (3 * data[data.columns[col]].std() )

        count = 0

        for entry in data[data.columns[col]]:

            if entry < min_t or entry > max_t:
                count += 1

        percentage = count / data.shape[0]
        outliers[data.columns[col]] = percentage * 100

        # print(f'Column: {col} has {percentage*100}% of outliers')

    # 4. Separate the features from the target data
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]

    # 5. Shuffle features and target together, then split into train, validation and testing
    shuffled_idx = X.sample(frac=1, random_state=42).index
    X_shuffle = X.loc[shuffled_idx]
    Y_shuffle = Y.loc[shuffled_idx]

    train_end = int(len(X_shuffle) * 0.6)
    valid_end = int(len(X_shuffle) * 0.8)

    x_train = X_shuffle.iloc[:train_end, :]
    x_valid = X_shuffle.iloc[train_end:valid_end, :]
    x_test = X_shuffle.iloc[valid_end:, :]

    y_train = Y_shuffle.iloc[:train_end]
    y_valid = Y_shuffle.iloc[train_end:valid_end]
    y_test  = Y_shuffle.iloc[valid_end:]

    # 6. Rescale features using training-set statistics only, to avoid leaking
    # validation/test information into the scaling
    train_mean = x_train.mean()
    train_std = x_train.std()

    x_train = (x_train - train_mean) / train_std
    x_valid = (x_valid - train_mean) / train_std
    x_test  = (x_test - train_mean) / train_std

    print(f'Train: {x_train.shape}, {y_train.shape}')
    print(f'Valid: {x_valid.shape}, {y_valid.shape}')
    print(f'Test: {x_test.shape}, {y_test.shape}')
    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == "__main__":

    x_train, x_valid, x_test, y_train, y_valid, y_test = dataloader()
    
    # Convert to torch tensors
    x_train = torch.tensor(x_train.values).float()
    # y_train = torch.tensor(y_train.values).float()
    y_train = torch.tensor(y_train.values).float().view(-1, 1)

    x_valid = torch.tensor(x_valid.values).float()
    # y_valid = torch.tensor(y_valid.values).float()
    y_valid = torch.tensor(y_valid.values).float().view(-1, 1)

    x_test = torch.tensor(x_test.values).float()
    # y_test = torch.tensor(y_test.values).float()
    y_test = torch.tensor(y_test.values).float().view(-1, 1)

    model = nn.Sequential(nn.Linear(x_train.shape[1], 10),
                          nn.ReLU(),
                          nn.Linear(10, 7),
                          nn.ReLU(),
                          nn.Linear(7, 5),
                          nn.ReLU(),
                          nn.Linear(5, 1))
    loss_fn = nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)

    epochs = 2500
    
    losses = []
    vlosses = []

    for epoch in range(epochs):

        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)
        vloss = loss_fn(y_valid, model(x_valid))
        print(f'Epoch: {epoch}, loss: {loss.item()}, validation loss: {vloss.item()}')
        losses.append(loss.item())
        vlosses.append(vloss.item())

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

y_test_pred = model(x_test)
print(f'Test loss: {loss_fn(y_test, y_test_pred)}')
print(f'Results: {(torch.hstack([y_test[:10],y_test_pred[:10]]))}')

fig, ax = plt.subplots()
ax.semilogy(range(epochs), losses, label='Training')
ax.semilogy(range(epochs), vlosses, label='Validation')
ax.grid()
ax.set_xlabel(r'Epochs')
ax.set_ylabel(r'Losses')
plt.savefig('plot.pdf', bbox_inches='tight')

