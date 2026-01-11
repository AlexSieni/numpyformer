from tensor import Tensor
import numpy as np

class Embedding:
    def __init__(self, fan_in, emb_size, symbol="Embedding"):
        self.fan_in = fan_in
        self.emb_size = emb_size
        self.symbol = symbol
        self.W = Tensor(np.random.randn(fan_in, emb_size), requires_grad=True)
    def __call__(self, x):
        # differentiate from positional embedding vs token embedding
        if x.ndim == 1: # position embedding
            out = Tensor(np.zeros((x.shape[0], self.emb_size)))
            for i in range(x.shape[0]):
                out.data[i] = self.W.data[x.data[i]]
            return out
        
        # x is of shape (batch_size, sequence_length) with integer indices
        batch_size, seq_length = x.shape
        out = Tensor(np.zeros((batch_size, seq_length, self.emb_size)))
        for i in range(batch_size):
            for j in range(seq_length):
                out.data[i, j] = self.W.data[x.data[i, j]]
        return out
    
    def params(self):
        return [self.W]
