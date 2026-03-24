import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    def row_softmax(X):
        x_max = np.amax(X, axis=1, keepdims=True)
        x_exp = np.exp(X - x_max)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    return row_softmax(Q @ K.T / (Q.shape[1] ** 0.5)) @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    Qs = np.array_split(Q, n_heads, axis=1)
    Ks = np.array_split(K, n_heads, axis=1)
    Vs = np.array_split(V, n_heads, axis=1)
    Rs = []
    for q, k, v in zip(Qs, Ks, Vs):
        Rs.append(self_attention(q, k, v))
    return np.concatenate(Rs, axis=1)