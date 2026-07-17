# Things that I've learned

1. Creating a single-layered neural network using PyTorch frameworks:
    a. Generating a input and target data set
    b. Defining the model using nn.Sequential()
    c. Defining the loss function (e.g. MSELoss())
    d. Defining the optimisation function (e.g. Adam's SGD or RMSProp)
    e. Writing the training loop
        
        i.    Get prediction
        ii.   Get loss
        iii.  Zero gradients! (Important!)
        iv.   Get gradients
        v.    Optimise parameters


# Things that I am unsure of

1. What loss function to use? (e.g. Mean squared error or Root-mean-squared errors?)
2. What optimisation function to use?


# Pointers for things that I am unsure of

1. Mean squared error is used for regression problems as it is generally smooth and differentiable.
2. Root-mean-square error introduces a scaling in the gradients which may be awkward, however the
values are of the same scale with the regression dataset so it may make more physical sense.
3. Other loss functions include (a) Cross entropy loss, (b) L1Loss (mean absolute error), (c)
(c) BCEWithLogitsLoss (Binary classification with logit loss)
4. Adam (Adaptive moment estimation) optimiser based on momentum with adaptable learning rates.
