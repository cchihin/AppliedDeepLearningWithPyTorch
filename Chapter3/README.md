# Things that I've learned

1. Trained a neural network for multi-class classification task to determine if 
a client would default (yes/no) on his credit card payments.

2. The softmax activation function converts numerical values into probabilities
where the sum of all output values is unity.

3. Combining the log-softmax with the negative least likelihood loss is typically 
used in classification task, reportedly equivalent to softmax activation + Cross 
Entropy loss.

4. Perform error analysis based on bias error (e.g. be = 0.15). By calculating 
bias error and variance error we can conclude if the model is over-/under-fitted.
For example, if bias error > variance error then the model is underfitted.

5. In the case where the model is underfitted (our example),
    a. Increase model complexity (more layers, more units)
    b. More data
    c. More training epochs

6. In the case where the model is overfitted
    a. Add L1/L2 regularisation
    b. Dropouts
    c. Simplify the model
    d. Early stop
    e. Ensemble methods

7. Here, the model is underfitted. We performed model selection by increasing number 
of `epochs` and also the network's `width` and `hidden layers`. When the model is 
adequatly fitted, we save the weights using `torch.save`, which can be loaded again 
using `checkpoint = torch.load(path)` and `model.load_state_dict(checkpoint['state_dict']`. 
Hence, it is important to know how to load the model class again.

# Things that I'm unsure off
1. The training notebooks included dropout layers which is absent from the answer booklet (what does the dropout layer do?)
2. Here I don't have to use `.view(-1,1)` on 1D outputs though, why?
3. I don't understand why log-softmax + NLLLoss is equivalent softmax + cross entropy loss.
4. The theory behind bayes error, which is used as 0.15?
5. In exporting the trace script, how can I export it as a module/function and what does check\_trace mean?


# Things that I've clarified
1. Adding dropout randomly switches off neuron activations, preventing the network
from overrelying on a single neuron. This is a regularisation technique that prevents
overfitting.
2. In this example, it is a multi-class classifcation problem where we only get
logits of `(N, num_classes)` with takes into account the target of `(N,)` through
`nn.CrossEntropyLoss()`
3. They are mathematically equivalent because `CrossEntropyLoss = -log(softmax(z))`
while `NLLLoss = -log p_c`. 
4. We are assuming Bayes error = 0.15 as the theorectical irreducible error.
