import torch
from typing import List, Tuple, Union
import numpy as np

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.float()
    return torch.tensor(data).float()

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels:   Union[List[float],      torch.Tensor],
    initial_weights: Union[List[float], torch.Tensor],
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> Tuple[List[float], float, List[float]]:
    """
    Train a single neuron (sigmoid activation) with mean-squared-error loss.

    Returns (updated_weights, updated_bias, mse_per_epoch)
    â weights & bias are rounded to 4 decimals; each MSE value is rounded too.
    """
    # Your implementation here
    X = to_tensor(features)
    y = to_tensor(labels).reshape(-1, 1)
    w = torch.nn.Parameter(to_tensor(initial_weights).reshape(-1, 1))
    b = torch.nn.Parameter(to_tensor(initial_bias).res