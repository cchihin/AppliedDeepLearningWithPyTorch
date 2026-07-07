import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Data pre-processing is usually performed in the following steps
# 1. Load the data
# 2. Missing entries - duplicate them if there are missing entries
# 3. Check for outliers - ignore them if the contribution is rather low
# 4. Check for class imbalance - resample them if they are of equally weightage
# 5. Standardise or normalise them

class Classifier(nn.Module):

    def __init__(self, input_size, hidden_dims):
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        for h_dim in hidden_dims:

            self.hidden_layers.append(nn.Linear(input_size, h_dim))
            input_size = h_dim

        self.output   = nn.Linear(input_size, 2)

    def forward(self, x):

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        out = F.log_softmax(self.output(x), dim=1)

        # out = F.softmax(self.output(z), dim=1)

        # Softmax converts the numerial values in probabilities where all entries sum to 1
        # LogSoftmax thats the logarithm of the softmax function [-\infty, 0]
        # LogSoftmax is typically combined with negative least likelihood loss which is related
        # to cross-entropy loss.

        return out

def datapreprocessor():

    # Loading excel data
    data = pd.read_excel('../Activity4/defaultofcreditcardclients.xls', skiprows=1)

    # Dropping irrelevant information
    data = data.drop(columns=['SEX', 'ID'])
    data.head()

    # Checking for missing values
    total = data.isnull().sum()
    percent = data.isnull().sum() / data.isnull().count() * 100
    missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])

    # Checking for outliers
    outliers = {}
    for i in range(data.shape[1]):

        right = data[data.columns[i]].mean() + 3 * data[data.columns[i]].std()
        left  = data[data.columns[i]].mean() - 3 * data[data.columns[i]].std()

        count = 0
        for entry in data[data.columns[i]]:
            if entry > right or entry < left:
                count += 1
        percent = count / data.shape[0] * 100
        outliers[data.columns[i]] = percent
        print(f'{percent:.2f}% outliers in {data.columns[i]}')

    # Notes
    # BILL_AMT1 and BILL_AMT4 has the largest amount of outliers at 2.3% 
    # However, it is still insignificant so its impact on the model is low


    # Checking for class imbalances
    # If the underlying target (e.g. yes/no) have more instances of defaults
    # than the other, the model will be biased accordingly.

    target = data["default payment next month"]
    yes = target[target == 1].count()
    no = target[target == 0].count()

    print(f'Yes: {yes / data.shape[0] * 100:.2f}%, no: {no / data.shape[0]*100:.2f}%')

    # Notes
    # There are 23364 entries of non-default customers and 6636 entries of default cusotmers
    # We need to balance the classes

    data_yes = data[data["default payment next month"] == 1]
    data_no = data[data["default payment next month"] == 0]

    over_sampling = data_yes.sample(no, replace=True, random_state=0)
    data_resampled = pd.concat([data_no, over_sampling], axis=0)


    # Splitting into targets and features and rescaling
    X = data_resampled.iloc[:,:-1]
    Y = data_resampled.iloc[:,-1]
    X_norm = (X - X.min()) / (X.max() - X.min())

    # Concatenating
    final_data = pd.concat([X_norm, Y], axis=1)
    final_data.to_csv('dccc_cleaned.csv', index=False)
    
    x_new, x_test, y_new, y_test = train_test_split(X_norm, Y, test_size=0.2,
                                                    random_state=0)

    dim = x_test.shape[0] / x_new.shape[0]

                                                
    x_train, x_valid, y_train, y_valid = train_test_split(x_new, y_new, test_size=dim,
                                                    random_state=0)
    print(f'Training sets: {x_train.shape}, {y_train.shape}')
    print(f'Validation sets: {x_valid.shape}, {y_valid.shape}')
    print(f'Testing sets: {x_test.shape}, {y_test.shape}')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == "__main__":

    x_train, x_valid, x_test, y_train, y_valid, y_test = datapreprocessor()

    # Converting them to torch tensors
    x_valid = torch.tensor(x_valid.values).float()
    y_valid = torch.tensor(y_valid.values)
    x_test = torch.tensor(x_test.values).float()
    y_test = torch.tensor(y_test.values).float()


    layers_list = [[100, 50], [100, 100, 50], [100, 100, 100, 50], [100, 100, 100, 100, 5]]
    epochs_list = [1000, 2000, 3000]

    # layers_list = [[10 for i in range(2)], [10 for i in range(3)]]
    # epochs_list = [5, 10]

    fig, ax = plt.subplots(len(layers_list), len(epochs_list))

    ii = 0
    for layers in layers_list:

        jj = 0
        for epochs in epochs_list:

            model = Classifier(x_train.shape[1], layers)

            loss_fn = nn.NLLLoss()

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # epochs = 50

            batch_size = 2**18
            
            train_losses, valid_losses, train_acc, valid_acc = [], [], [], []

            for e in range(epochs):

                x, y = shuffle(x_train, y_train)

                running_loss = 0
                running_acc = 0
                iterations = 0

                for i in range(0, len(x), batch_size):
                    iterations += 1
                    b = i + batch_size
                    x_batch = torch.tensor(x.iloc[i:b,:].values).float()
                    y_batch = torch.tensor(y.iloc[i:b].values)
                    
                    pred = model(x_batch)

                    loss = loss_fn(pred, y_batch)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                    ps = torch.exp(pred)
                    top_p, top_class = ps.topk(1, dim=1)
                    running_acc += accuracy_score(y_batch, top_class)

                valid_loss = 0
                acc = 0

                with torch.no_grad():
                    valid_pred = model(x_valid)
                    valid_loss = loss_fn(valid_pred, y_valid)
                    valid_ps = torch.exp(valid_pred)
                    valid_top_p, valid_top_class = valid_ps.topk(1, dim=1)
                    acc = accuracy_score(y_valid, valid_top_class)


                train_losses.append(running_loss / iterations)
                train_acc.append(running_acc / iterations)
                valid_losses.append(valid_loss)
                valid_acc.append(acc)

                print(f'Epoch: {e+1}/{epochs}')
                print(f'Training Loss: {running_loss/iterations:.3f}')
                print(f'Training Accuracy: {running_acc/iterations:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')
                print(f'Validation Accuracy: {acc:.3f}')
                print(f'')

            ax[ii][jj].plot(range(epochs), train_acc, '-o', label='Train')
            ax[ii][jj].plot(range(epochs), valid_acc, '-o', label='Valid')
            ax[ii][jj].set_xlabel(r'Epochs')
            ax[ii][jj].set_ylabel(r'Accuracy (%)')
            ax[ii][jj].grid()
            ax[ii][jj].legend()

            # Post-processing
            training_err = 1 - running_acc/iterations
            validation_err = 1 - acc

            bayes_err = 0.15
            bias = training_err - bayes_err
            variance = validation_err - training_err

            ax_title = f'Epochs= {epochs}, ' + fr'$N_{{layers}}= {len(layers)}$' + '\n' + \
            f'bias= {bias:.3f}, var= {variance:.4f}'

            ax[ii][jj].set_title(ax_title)

            # Iterating over epochs_list
            jj += 1

        # Iterating over increasing hidden layers
        ii += 1

    fig.set_size_inches(len(epochs_list)*6, len(layers_list)*4)
    fig.subplots_adjust(hspace=0.4)
    fig.savefig('model_selection.pdf', bbox_inches='tight')
