# Things that I've learned

1. Trained a neural network for multi-class classification task to determine if a client would default (yes/no) on his credit card payments.

2. The softmax activation function converts numerical values into probabilities where the sum of all output values is unity.

3. Combining the log-softmax with the negative least likelihood loss is typically used in classification task, reportedly equivalent to softmax activation + Cross Entropy loss.

# Things that I'm unsure off
1. The training notebooks included dropout layers which is absent from the answer booklet (what does the dropout layer do?)
2. Here I don't have to use `.view(-1,1)` on 1D outputs though, why?
3. I don't understand why log-softmax + NLLLoss is equivalent 2. Here I don't have to use `.view(-1,1)` on 1D outputs though, why?
4. I don't understand why log-softmax + NLLLoss is equivalent softmax + cross entropy loss.
