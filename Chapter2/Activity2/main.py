import pandas as pd

def dataloader():

    # 1. Load csv file
    data = pd.read_csv('YearPredictionMSD.txt', header=None, nrows=50000)

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

        print(f'Column: {col} has {percentage*100}% of outliers')

    # 4. Separate the features from the target data
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]

    # 5. Rescale features
    X_rescale = (X - X.mean()) / (X.std())

    # 6. Split the dataset into train, validation and testing
    X_shuffle = X_rescale.sample(frac=1)
    Y_shuffle = Y.sample(frac=1)
    train_end = int(len(X_rescale) * 0.6)
    valid_end = int(len(X_rescale) * 0.8)

    x_train = X_shuffle.iloc[:train_end, :]
    x_valid = X_shuffle.iloc[train_end:valid_end, :]
    x_test = X_shuffle.iloc[valid_end:, :]

    y_train = Y_shuffle.iloc[:train_end]
    y_valid = Y_shuffle.iloc[train_end:valid_end]
    y_test  = Y_shuffle.iloc[valid_end:]

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == "__main__":

    x_train, x_valid, x_test, y_train, y_valid, y_test = dataloader()
