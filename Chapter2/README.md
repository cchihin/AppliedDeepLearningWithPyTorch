# Things that I've learned
1. Pre-processing datasets that generally involves

    a. Loading the raw data file (e.g. using Pandas)
    b. Checking for missing data (e.g. df.isnull().sum())
    c. Checking of outliers (e.g. Based on 3 times of std as threshold)
    d. Splitting dataset into inputs and targets
    e. Rescaling inputs (e.g. Normalisation v.s. standardisation)
    f. Splitting inputs and targets into train/validation/test datasets.

2. Pre-processing a dataset and then feeding it into a training loop

3. For 1D targets, Claude appended `.view(-1,1)` to the target variables leading to faster and correct training. 


# Things that I'm unsure off
1. What do I do with missing data? (fill them up with duplicates?)
2. How to I define the outliers - is 3 times of std sufficient?
3. What is reccommended train/validate/test split ratio? 60%/20%/20%
4. Why do I rescale and how do I choose between normalisation/standardisation (e.g. rescaling so that the raw outputs are do become biased by their large numerical values)
5. How do I choose between normalisation or standardisation during re-scaling?
6. What does `.view(-1,1)` really do?

# Follow-up
1. Drop first, or just fill with mean or median
2. Good enough for outlier detection, IQR method may benefit
3. 60/20/20 is fine for small to medium datasets, but I will adopt 90/5/5 or 98/1/1 splits
for larger datasets.
4. Standardisation is usually the preferred choice. In cases that the max and min
values are known (e.g. 0-255 for RGB images), normalisation could be adopted.
5. Converts broadcasts a tensor from (N,) to (N,1), where N refers to the batch/sample size.
This usually occurs when we want to compare the output of (N, 1) from a nn.Linear(128, 1) 
from a target of (N,), then we need to transform the target from (N,) to (N,1) to match
the shape of the output, (N,1).
