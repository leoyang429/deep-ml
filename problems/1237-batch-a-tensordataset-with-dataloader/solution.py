import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    num_batches = 0
    first_batch_X_shape_tuple = None
    for x, target in dataloader:
        num_batches += 1
        if first_batch_X_shape_tuple is None:
            first_batch_X_shape_tuple = tuple(x.shape)
    return (num_batches, first_batch_X_shape_tuple)
