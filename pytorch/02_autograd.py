# %%
# Section 1: Fundamentals of automatic differention and PyTorch's Autograd

# Subsection 1.1: A quick motivation background
# (No code blocks in this subsection)

# Subsection 1.2: What Do We Need Autograd For?

# %%
# If you don't have SymPy install, you need to install it first:
# !pip install sympy      # uncomment this statement to install the package sympy
import sympy as sp
import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# Section 2: A very simple example

# %%
x = torch.linspace(0, 2*np.pi, 20, requires_grad=True)
y = torch.sin(x)

plt.figure(figsize=(4, 2.8))
plt.plot(x.detach(), y.detach()), plt.grid(), plt.show()
print(y)

# %%
# Subsection 2.1: The backward operation
z = y.sum()
z.backward()
print(x.grad)

# %%
plt.plot(x.detach(), y.detach(), label='sin(x)')
plt.plot(x.detach(), x.grad.detach(), label='gradient of sin(x)')
plt.legend()
plt.show()

# %%
# Section 3: More complex example

# %%
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2 + 2*x + 1
z = y.sum()

z.backward()
print(f"Gradients w.r.t x: {x.grad}")

# %%
# Section 4: Autograd in training

# %%
# Subsection 4.1: Defining the model
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# %%
# Subsection 4.2: Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%
# Subsection 4.3: Training loop (one step)
# Input data
inputs = torch.tensor([[1.0], [2.0]], requires_grad=True)
labels = torch.tensor([[2.0], [4.0]])

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, labels)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss after one step: {loss.item()}")