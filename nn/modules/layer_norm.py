import numpy as np
from tensor import Tensor, as_tensor

class LayerNorm:
    def __init__(self, fan_in, eps=1e-5, symbol=None):
        self.symbol = symbol
        self.eps = eps

        # scale and shift
        self.gamma = Tensor(np.ones(fan_in))
        self.beta  = Tensor(np.zeros(fan_in))

    def forward(self, x: Tensor):
        mean = x.data.mean(axis=-1, keepdims=True)
        var  = ((x.data - mean) ** 2).mean(axis=-1, keepdims=True)
        std  = np.sqrt(var + self.eps)

        x_hat = (x.data - mean) / std
        out_data = self.gamma.data * x_hat + self.beta.data
        out = Tensor(out_data, symbol=self.symbol)

        out.parents = (x, self.gamma, self.beta)

        def _backward():
            g = out.grad  # (…, D)

            # gradients for gamma & beta
            self.gamma.grad += np.sum(g * x_hat, axis=tuple(range(g.ndim - 1)))
            self.beta.grad  += np.sum(g, axis=tuple(range(g.ndim - 1)))

            # gradient w.r.t input
            N = x_hat.shape[-1]
            dx = (1 / std) * (
                g * self.gamma.data
                - np.mean(g * self.gamma.data, axis=-1, keepdims=True)
                - x_hat * np.mean(g * self.gamma.data * x_hat, axis=-1, keepdims=True)
            )

            if x.requires_grad:
                x.grad += dx

        out._backward = _backward
        return out

    def params(self):
        return [self.gamma, self.beta]
