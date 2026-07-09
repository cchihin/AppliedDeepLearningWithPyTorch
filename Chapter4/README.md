# What have I learned?

1. Compute the output size of a convolutional network
    
    a. 1st convolution layer
        Input: (256, 256, 3)
        Convolution layer: 16 filters, 3 widths, 1 stride and pad
        Output: (256, 256, 16)

    b. 1st Pooling layer
        Input: (256, 256, 16)
        Pooling layer: 2 widths and stride 
        Output: (128, 128, 16)

    c. 2nd convolution layer
        Input: (128, 128, 16)
        Convolution layer: 8 filters, 7 widths, 1 stride and 3 pads
        Output: (128, 128, 8)

    d. 2st Pooling layer
        Input: (128, 128, 8)
        Pooling layer: 2 widths and stride 
        Output: (64, 64, 8)
    
