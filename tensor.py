# Goal is to create mini version of transformer architecture
# scope will be numpy only

### System design
# Layers- Linear, Norm, Attention head, 
# Combo - Multi attention head, Feed forward
# Model - Overall model build
# Also include skip connections, kai ming init, low weight for last layer

from typing import Any
import numpy as np

class Tensor:
    def __init__(self, data, symbol=None, requires_grad=True, dtype: Any = np.float32): # symbol for debug
        self.data = np.array(data, dtype=dtype)
        self.symbol = symbol
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.parents = ()
        self._backward = lambda: None

    def backward(self):
        # Build a topo and recursively backward through parents
        visited = set()
        topo = list()
        def build_topo(ten: Tensor):
            # print(ten.symbol, [par.symbol for par in ten.parents])
            if ten not in visited:
                visited.add(ten)
                for parent in ten.parents:
                    build_topo(parent)
                topo.append(ten)
        build_topo(self)
        for ten in reversed(topo):
            ten._backward()
            
    def __hash__(self):
        return id(self)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    # cast to array when array operation is called on tensor
    def __array__(self, dtype:Any=None):
        return self.data.astype(dtype) if dtype else self.data
    
    def view(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, dtype=self.data.dtype)
        out.parents = (self, )

        def _backward():
            if self.requires_grad:
                assert out.grad is not None
                assert self.grad is not None
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out



    #### Tensor operations, numpy wrapped

    def __matmul__(self, other):
        other = as_tensor(other, requires_grad=False)
    
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
    
        out.parents = (self, other)
    
        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None

            if self.requires_grad:
                grad_self = out.grad @ other.data.swapaxes(-1, -2)
                grad_self = _debroadcast(grad_self, self.data.shape)
                assert self.grad is not None
                self.grad += grad_self

            if other.requires_grad:
                grad_other = self.data.swapaxes(-1, -2) @ out.grad
                grad_other = _debroadcast(grad_other, other.data.shape)
                assert other.grad is not None 
                other.grad += grad_other

    
        out._backward = _backward
        return out

    def __add__(self, other):
        other = as_tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = (self, other)
    
        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None

            if self.requires_grad:
                self.grad += _debroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _debroadcast(out.grad, other.shape)
    
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out.parents = (self, )
        
        def _backward():
            if self.requires_grad:
                assert out.grad is not None
                assert self.grad is not None
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        """
        Called when self is a tensor, other may be a tensor
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Called when other is a tensor, but self is not
        """
        return as_tensor(self, requires_grad=False) - other 
    
    def __mul__(self, other):
        other = as_tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = (self, other)

        def _backward():
            if not out.requires_grad:
                return
            
            assert out.grad is not None
            if self.requires_grad:
                self.grad += _debroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _debroadcast(out.grad * self.data, other.shape)
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = as_tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out.parents = (self, other)

        def _backward():
            if not out.requires_grad:
                return
            
            assert out.grad is not None
            if self.requires_grad:
                self.grad += _debroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad += _debroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
        
        out._backward = _backward
        return out
    
    def __pow__(self, other: float):
        out = Tensor(self.data**other, requires_grad=self.requires_grad)
        out.parents = (self,)

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None
            assert self.grad is not None
            # power rule
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None
            assert self.grad is not None
            self.grad += (1 / self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=-1, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None
            assert self.grad is not None
            self.grad += out.grad / self.shape[axis]
        
        out._backward = _backward
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if self.requires_grad:
                assert out.grad is not None
                assert self.grad is not None
                self.grad += 0.5 * out.grad / np.sqrt(self.data)
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if self.requires_grad:
                assert out.grad is not None
                assert self.grad is not None
                # inverse transpose
                self.grad += out.grad.transpose(*np.argsort(axes))
        out._backward = _backward
        return out

    def max(self, axis=-1, keepdims=False):
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None

            if self.requires_grad:
                # gradient flows only to max elements
                grad_self = np.zeros_like(self.data)
                if keepdims:
                    mask = (self.data == out.data)
                else:
                    expanded_out = np.expand_dims(out.data, axis=axis)
                    mask = (self.data == expanded_out)
                grad_self += mask * out.grad
                self.grad += grad_self

        out._backward = _backward
        return out
    
    def sum(self, axis=-1, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out.parents = (self, )

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None

            if self.requires_grad:
                assert self.grad is not None
                grad_self = out.grad
                if not keepdims:
                    grad_self = np.expand_dims(grad_self, axis=axis)
                grad_self = np.broadcast_to(grad_self, self.data.shape)
                self.grad += grad_self

        out._backward = _backward
        return out

    @staticmethod
    def where(condition, x_if, x_else):
        condition = np.array(condition)
        x_if = as_tensor(x_if, requires_grad=False)
        x_else = as_tensor(x_else, requires_grad=False)

        out = Tensor(np.where(condition, x_if.data, x_else.data),
                     requires_grad=x_if.requires_grad or x_else.requires_grad)
        out.parents = (x_if, x_else)

        def _backward():
            if not out.requires_grad:
                return
            assert out.grad is not None

            if x_if.requires_grad:
                grad_if = np.where(condition, out.grad, 0)
                x_if.grad += _debroadcast(grad_if, x_if.shape)

            if x_else.requires_grad:
                grad_else = np.where(condition, 0, out.grad)
                x_else.grad += _debroadcast(grad_else, x_else.shape)

        out._backward = _backward
        return out
    

    def cross_entropy(self, y):
        """
        Compute cross entropy loss.
        self: logits of shape (N, C) where N is batch size, C is number of classes
        y: targets of shape (N,) with integer class labels
        """
        
        B = self.shape[0]
        logits = self.data
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss_value = -np.log(probs[np.arange(B), y.data]).mean()
        
        out = Tensor(loss_value, requires_grad=self.requires_grad)
        out.parents = (self,)

        def _backward():
            if not self.requires_grad:
                return
            assert out.grad is not None
            assert self.grad is not None
            
            grad = probs.copy()
            grad[np.arange(B), y] -= 1
            grad /= B
            self.grad += grad

        out._backward = _backward
        return out

    def debug(self, name=""):
        print(f"\n--- Tensor Debug {name} ---")
        print("id:", id(self))
        print("data:", self.data)
        print("shape:", self.data.shape)
        print("grad:", self.grad)
        print("requires_grad:", self.requires_grad)
        print("parents:", self.parents)
        print("num_parents:", None if self.parents is None else len(self.parents))
        print("backward fn:", self._backward)


def as_tensor(data, requires_grad=True):
    if isinstance(data, Tensor):
        return data
    return Tensor(data, requires_grad=requires_grad)

def _debroadcast(grad: np.ndarray, shape: tuple):
    # for additional dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # sum over broadcasted axes
    for dim, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=dim, keepdims=True)

    return grad
    

# A = np.random.randn(1)
# B = np.random.randn(5, 5)
# print(A, B, A*B)
