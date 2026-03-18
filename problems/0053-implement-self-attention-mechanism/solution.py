import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    
    def row_softmax(X):
        x_max = np.amax(X, axis=1, keepdims=True)
        x_exp = np.exp(X - x_max)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    return row_softmax(Q @ K.T / (Q.shape[1] ** 0.5)) @ V