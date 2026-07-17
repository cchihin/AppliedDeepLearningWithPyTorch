# What have I learned?

1. Compute the output size (e.g. shape and dimension) of a convolutional network
	
    a. Convolution layer formula
        Output width = (Input width - filter width + 2 * padding) / Stride + 1   
        e.g (3, 1, 1) retains the same width

    b. Pooling layer formula
        Output width = (Input width - filter width) / stride + 1
        e.g. (2, 2) halves the width

2. Training the CNN Classifier using CIFAR10 by changing batch size and increasing dropout

    a. Dropout (at a given probability) removes neurons from a layer. It is a
        form of regulariser that prevents overfitting. (e.g. can be used when
        the validation error starts to increase)

    b. Lower the batch size leads to more weight updates which may accelerate 
        convergence. The degree of convergence is likely to depend on the 
        quality of the dataset. If the batches do not differ too much, then
        lower batch sizes will quicker convergence.
        
    c. Decreasing batch sizes (or increase weight updates) is akin to increasing 


        learning rate.

3. Overfitting vs underfitting
    
    a. Underfitting occurs is when the model is unable to reduce both the training
        validation error any further. The reason for this is usually the model (
        e.g. neural network) isn't powerful enough (under capacity) to capture the
        underlying complexities of the model.

    b. Overfitting occurs when the training error continue to decrease while validation
        error increases. The occurs because the model is so powerful that it fits the 
        training dataset, failing to generalise to a more diverse dataset such as
        the validation dataset.

4. Data augmentation augments the dataset by applying flips and noise, generalising the 
model.

5. Batch normalisation normalises the output from each layer, and was reported to have
lower losses and higher accuracy in the CIFAR10 dataset.

# What I am unsure of
1. How to select the number of filter widths, pooling and channels of CNNs?

2. What do the arguments in BatchNorm refer to?

3. What do the arguments in the Data Augmentation mean?


# Follow-up
1. Filter width of 3x3 is the default choice since it gives the same receptive field
as 5x5 shown in the VGG dataset.

2. Intuition is to increase channels as spatial resolution decreases. Channels are
engineered to detect features so in the earlier layers, the features are larger and more
sparse and as the spatial resolution decreases, the combination of features increase.

3. Pooling 2x2 halves the spatial resolution which is the standard.

4. BatchNorm referes to normalisation over features (e.g. channels) with learning 
scaling and shifting parameters via a user-defined affine value.

5. Data augmentation makes the model more invariant under flips and rotations.

