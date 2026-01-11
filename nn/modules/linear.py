import numpy as np
from tensor import Tensor, as_tensor

class Linear:
    def __init__(self, fan_in, fan_out, symbol=None):
        self.W = Tensor(np.random.randn(fan_in, fan_out) * .1)
        self.b = Tensor(np.random.randn(fan_out) * .1)
        self.symbol = symbol
    def forward(self, x: Tensor):
        self.out = Tensor(x.data @ self.W.data + self.b.data, symbol=self.symbol)
        return self.out
    def params(self):
        return [self.W, self.b]
