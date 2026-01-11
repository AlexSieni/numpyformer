import numpy as np
from tensor import Tensor, as_tensor

class LayerNorm:
    def __init__(self, fan_in, eps=1e-5, symbol=None):
        self.symbol = symbol
        self.eps = eps

        # scale and shift
        self.gamma = Tensor(np.ones(fan_in) * .1)
        self.beta  = Tensor(np.zeros(fan_in) * .1)

    def forward(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)
        var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std  = (var + self.eps).sqrt()

        x_hat = (x - mean) / std
        out_data = self.gamma.data * x_hat + self.beta
        self.out = Tensor(out_data, requires_grad=True)
        return self.out

    def params(self):
        return [self.gamma, self.beta]
