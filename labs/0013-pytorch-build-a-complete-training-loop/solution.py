# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    history = []
    
    # Your code here
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    
    n_samples = len(X_train)
    batch_num = (n_samples + batch_size - 1) // batch_size
    for e in range(epochs):
        perm = torch.randperm(n_samples)
        X_shuffle, y_shuffle = X_train[perm], y_train[perm]
        train_loss = 0
        model.train()
        for b in range(batch_num):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_samples)
            X, y = X_shuffle[start:end], y_shuffle[start:end]
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= batch_num
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            loss = criterion(pred, y_val)
            val_loss = float(loss.item())
            val_accuracy = (pred.argmax(dim=1) == y_val).float().mean().item()
            history.append({
                'epoch': e + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
    
    return history