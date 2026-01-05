import numpy as np
from tensor import Tensor, as_tensor

class Tanh:
    def __init__(self, symbol=None):
        self.symbol = None
    def forward(self, x: Tensor):
        self.out = (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
        self.out = Tensor(self.out, symbol=self.symbol)
        self.out.parents = (x,)
        def _backward():
            x.grad += (1 - (self.out.data)**2) * self.out.grad
        self.out._backward = _backward
        return self.out
    def params(self):
        return []
