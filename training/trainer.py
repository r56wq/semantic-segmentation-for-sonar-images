import torch
from torch import nn
import time
import matplotlib.pyplot as plt

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 
               devices=None, plot_graph=True, print_time=True):
    """
    Train a model with multiple GPUs.
    
    Args:
        net: Neural network model
        train_iter: Training data iterator
        test_iter: Test data iterator
        loss: Loss function
        trainer: Optimizer
        num_epochs: Number of epochs
        devices: List of GPU devices (default: try all GPUs)
        plot_graph: Boolean to plot training graph
        print_time: Boolean to print training time
    """
    if devices is None:
        devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    
    # Initialize timing and batches
    start_time = time.time()
    num_batches = len(train_iter)
    
    # Prepare for plotting if enabled
    if plot_graph:
        train_losses, train_accs, test_accs = [], [], []
    
    # Move model to multiple GPUs if available
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    for epoch in range(num_epochs):
        # Metrics: train_loss_sum, train_acc_sum, num_examples, num_predictions
        metrics = [0.0, 0.0, 0, 0]
        
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metrics[0] += l.item()
            metrics[1] += acc
            metrics[2] += labels.shape[0]
            metrics[3] += labels.numel()
            
            # Update plot if enabled
            if plot_graph and (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_losses.append(metrics[0] / metrics[2])
                train_accs.append(metrics[1] / metrics[3])
                test_accs.append(None)
        
        # Calculate test accuracy
        test_acc = evaluate_accuracy(net, test_iter, devices[0])
        if plot_graph:
            train_losses.append(None)
            train_accs.append(None)
            test_accs.append(test_acc)
        
        # Print results
        print(f'Epoch {epoch + 1}: loss {metrics[0] / metrics[2]:.3f}, '
              f'train acc {metrics[1] / metrics[3]:.3f}, test acc {test_acc:.3f}')
    
    # Final metrics
    if print_time:
        total_time = time.time() - start_time
        print(f'{metrics[2] * num_epochs / total_time:.1f} examples/sec on {str(devices)}')
    
    # Plot if enabled
    if plot_graph:
        epochs = [i/num_batches + e for e in range(num_epochs) for i in range(num_batches)]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(train_losses)], train_losses, label='train loss')
        plt.plot(epochs[:len(train_accs)], train_accs, label='train acc')
        plt.plot(range(1, num_epochs + 1), test_accs[::num_batches], label='test acc')
        plt.xlabel('epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """
    Train for a minibatch with multiple GPUs.
    """
    # Move data to device
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    
    # Forward and backward pass
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    
    # Calculate metrics
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    
    return train_loss_sum, train_acc_sum

def accuracy(pred, y):
    """Calculate accuracy."""
    pred_classes = pred.argmax(dim=1)
    correct = (pred_classes == y).float().sum()
    return correct.item()

def evaluate_accuracy(net, data_iter, device):
    """Evaluate accuracy of the model."""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            pred = net(X)
            correct += accuracy(pred, y)
            total += y.numel()
    return correct / total

# Example usage:
"""
# Assuming you have your model, data loaders, and loss function defined
model = YourModel()
train_loader = YourTrainLoader()
test_loader = YourTestLoader()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_ch13(model, train_loader, test_loader, loss_fn, optimizer, num_epochs=10,
          plot_graph=True, print_time=True)
"""