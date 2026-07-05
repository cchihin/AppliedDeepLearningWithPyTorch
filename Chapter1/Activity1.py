# 1. Import the necessary libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 2. Create a dummy input x and target y
dummy_input = torch.randn(100, 5)
target = torch.randint(0, 2, (100, 1)).type(torch.FloatTensor)


# 3. Define the architecture and craete the model
# Output = sig(W_2 * x_2 + b_2)
# x_2 = ReLU(W_1 * x_0 + b_1)
hidden_dim = 10
model = nn.Sequential(nn.Linear(5, 1),
                      nn.Sigmoid())


# 4. Defining the loss function [What's the difference between MSE loss?]
loss_fn = nn.MSELoss()

# 5. Defining the optimisation procedure [What optimisation should I use?]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Run the training step
epochs = 100
losses = []
for epoch in range(epochs):

    # Call the model
    y_pred = model(dummy_input)

    # Compute the loss
    loss = loss_fn(y_pred, target)
    losses.append(loss.item())

    # Zero the gradients (since grads accumulate)
    optimizer.zero_grad()

    # Calculate the gradients of the loss function
    loss.backward()

    # Update the optimizer
    optimizer.step()
    print(f'Epoch: {epoch}, loss: {loss.item()}')

fig, ax = plt.subplots()
ax.plot(range(epochs), losses)
ax.grid()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

plt.show()
