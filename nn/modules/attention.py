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
        B, T, C = x.shape

        # causal mask
        tril = np.tril(np.ones((T, T)))

        # projections
        Q = x @ self.Wq   # (B, T, h)
        K = x @ self.Wk   # (B, T, h)
        V = x @ self.Wv   # (B, T, h)

        # attention scores
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])
        

        # apply causal mask
        scores = np.where(tril, scores, -1e9)

        # softmax
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # weighted sum
        out = weights @ V   # (B, T, h)
        out = Tensor(out)
        out.children = ()
        return out


# -----------------------------
# Test attention head
# -----------------------------
a = AttentionHead(emb_size=6)
x = np.random.randn(2, 4, 6)
out = a.forward(x)
print("Attention output shape:", out.shape)
