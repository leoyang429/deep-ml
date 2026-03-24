import numpy as np
import math

def pos_encoding(position: int, d_model: int):
    # Your code here
    pos_encoding = [math.sin(pos / (10000 ** (i / d_model))) if i % 2 == 0 else \
                    math.cos(pos / (10000 ** ((i - 1) / d_model))) for pos in range(position) for i in range(d_model)]
    return np.array(pos_encoding, dtype=np.float16).reshape(position, d_model)