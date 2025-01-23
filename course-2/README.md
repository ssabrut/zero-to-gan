# Working with Images: PyTorch Image Classification & Logistic Regression

This notebook covers working with image data in PyTorch, implementing logistic regression for MNIST digit classification.

## Key Topics Covered

## 1. Image Data Loading
- Loading MNIST dataset using torchvision
- Understanding image tensors and transformations
- Working with training, validation and test splits

## 2. Data Preprocessing
- Converting images to tensors
- Normalizing pixel values
- Creating data loaders for batch processing

## 3. Logistic Regression Implementation
- Building a logistic regression model for image classification
- Understanding model architecture for MNIST
- Working with high-dimensional inputs (28x28 images)

## 4. Training & Evaluation
- Implementing training loops
- Computing accuracy metrics
- Using cross-entropy loss for classification
- Monitoring model performance

## Code Examples

## Loading MNIST Dataset
```python
from torchvision.datasets import MNIST

# Load training data
train_dataset = MNIST(root='data/', download=True)

# Load test data
test_dataset = MNIST(root='data/', train=False)
```

## Data Preprocessing
```python
# Convert to tensors
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
```

## Logistic Regression Model
```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.linear(x)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
```

## Training Loop
```python
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    
    for epoch in range(epochs):
        # Training
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history
```

## Dataset Used
The example uses the MNIST dataset:
- 60,000 training images
- 10,000 test images 
- 28x28 grayscale images of handwritten digits (0-9)
- 10 classes for digit classification

## Learning Outcomes
- Loading and preprocessing image data in PyTorch
- Implementing logistic regression for image classification
- Working with high-dimensional inputs
- Training and evaluating classification models
- Using cross-entropy loss and accuracy metrics
- Monitoring training progress with validation
