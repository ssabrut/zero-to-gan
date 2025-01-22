# Basics & Gradient Descent: Tensors, Autograd & Linear Regression

This notebook covers the fundamentals of PyTorch, including tensors, autograd, and implementing linear regression from scratch.

## Key Topics Covered

## 1. Tensors
- Creating tensors of different dimensions (scalar, vector, matrix, 3D)
- Tensor operations and manipulations
- Understanding tensor shapes and data types

## 2. Gradient Descent
- Computing gradients using autograd
- Understanding backpropagation
- Implementing gradient descent manually

## 3. Linear Regression Implementation
- Creating a basic linear regression model from scratch
- Implementing loss functions (MSE)
- Training the model using gradient descent

## 4. PyTorch Built-in Features
- Using `torch.nn.Linear` for linear regression
- Working with `DataLoader` and `TensorDataset`
- Implementing optimizers (SGD)

## Code Examples

## Creating Tensors
```python
import torch

# Scalar tensor
t1 = torch.tensor(1.)

# Vector tensor
t2 = torch.tensor([1,2.,3,4])

# Matrix tensor
t3 = torch.tensor([
    [1.,2,3],
    [4,5,6]
])
```

## Basic Linear Regression
```python
# Model
def linear_regression(x):
    return x @ w.t() + b

# Loss function
def mean_squared_error(y_pred, y_true):
    diff = y_pred - y_true
    return torch.sum(diff ** 2) / diff.numel()

# Training loop
for epoch in range(epochs):
    yhat = linear_regression(inputs)
    loss = mean_squared_error(yhat, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
```

## Using PyTorch's Built-in Features
```python
# Create model
model = nn.Linear(3,2)

# Define loss function
loss_fn = F.mse_loss

# Create optimizer
opt = SGD(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(epochs):
    for x, y in train_dl:
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

## Dataset Used
The example uses a synthetic dataset with:
- Features: temperature, rainfall, humidity
- Targets: apple and orange yield predictions

## Learning Outcomes
- Understanding tensor operations and gradients in PyTorch
- Implementing linear regression from scratch
- Using PyTorch's built-in modules for efficient implementation
- Working with datasets and dataloaders
- Implementing training loops with gradient descent