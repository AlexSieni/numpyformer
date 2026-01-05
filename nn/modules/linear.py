import numpy as np
from tensor import Tensor, as_tensor

class Linear:
    def __init__(self, fan_in, fan_out, symbol=None):
        self.W = Tensor(np.random.randn(fan_in, fan_out))
        self.b = Tensor(np.random.randn(fan_out))
        self.symbol = symbol
    def forward(self, x: Tensor):
        # x has shape (#samples, fan_in (num features))
        # (#sam, fan_in) @ (fan_in, fan_out) + (fan_out)
        self.out = Tensor(x.data @ self.W.data + self.b.data, symbol=self.symbol)
        self.out.parents = (self.W, self.b, x)
        def _backward():
            # out has shape (#samples, fan_out)
            self.W.grad += x.data.T @ self.out.grad
            if x.requires_grad:
                x.grad += self.out.grad @ self.W.data.T
            # (#sam, fan_out) + (, fan_out) = broadcast for #sam rows
            self.b.grad += self.out.grad.sum(axis=0)
        self.out._backward = _backward
        return self.out
    def params(self):
        return [self.W, self.b]
