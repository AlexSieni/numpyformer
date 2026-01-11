import numpy as np
from tensor import Tensor, as_tensor

class AttentionHead:
    def __init__(self, emb_size, head_size=None):
        # Xavier / fan-in scaling to keep variance stable
        if head_size == None:
            head_size = emb_size
        self.Wq = Tensor(np.random.randn(emb_size, head_size) / np.sqrt(emb_size))
        self.Wk = Tensor(np.random.randn(emb_size, head_size) / np.sqrt(emb_size))
        self.Wv = Tensor(np.random.randn(emb_size, head_size) / np.sqrt(emb_size))

    def forward(self, x):
        """
        x: (B, T, C)
        """
        x = as_tensor(x)
        B, T, C = x.shape

        # causal mask
        tril = Tensor(np.tril(np.ones((T, T))))

        # projections
        Q = x @ self.Wq   # (B, T, h)
        K = x @ self.Wk   # (B, T, h)
        V = x @ self.Wv   # (B, T, h)

        # attention scores
        # scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])
        scores = (Q @ K.transpose(0, 2, 1)) * (K.shape[-1] ** -0.5)  # (B, T, T)
        

        # apply causal mask
        scores = Tensor.where(tril, scores.data, -1e9)

        # softmax
        exp_scores = (scores - scores.max(axis=-1, keepdims=True)).exp()
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # weighted sum
        self.out = weights @ V   # (B, T, h)
        return self.out
    
    def params(self):
        return [self.Wq, self.Wk, self.Wv]
